import numpy as np
import torch
from transformers import CLIPTokenizer

from diffusion.convert_weights import load_all
from diffusion.samplers.ddpm import DDPM
from diffusion.samplers.ddim import DDIM
from diffusion.samplers.euler import Euler
from diffusion.model.autoencoder import Autoencoder
from diffusion.model.clip import CLIP
from diffusion.model.unet import UNet


class InferencePipeline:
    def __init__(
        self,
        ckpt_path,
        device: torch.device | str,
        idle_device: torch.device | str,
        n_step_train: int = 1000
    ):
        self.device = device
        self.idle_device = idle_device
        self.n_step_train=n_step_train

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vae = Autoencoder()
        self.clip = CLIP()
        self.unet = UNet()

        load_all(ckpt_path, vae=self.vae, clip=self.clip, unet=self.unet)

        self.sampler_names = {
            "ddpm": DDPM,
            "ddim": DDIM,
            "euler": Euler
        }

    def _make_generator(self, seed: int, device: torch.device | str) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator
    
    def _make_sampler(self, sampler_name: str, n_step_inf):
        if sampler_name not in self.sampler_names:
            raise ValueError("Unknown sampler")
        return self.sampler_names[sampler_name](
            n_step_train=self.n_step_train,
            n_step_inf=n_step_inf
        )

    def _load_model(self, model: torch.nn.Module) -> None:
        model.to(device=self.device)
        model.eval()

    def _offload_model(self, model: torch.nn.Module) -> None:
        model.to(device=self.idle_device)

    def _rescale(
        self, x: torch.Tensor, old_min, old_max, new_min, new_max
    ) -> torch.Tensor:
        x -= old_min
        x *= (new_max - new_min) / (old_max - old_min)
        x += new_min
        return torch.clamp(x, new_min, new_max)

    def _postprocess_image(self, images):
        images = self._rescale(images, -1, 1, 0, 255)
        images = torch.permute(images, (0, 2, 3, 1))
        images = images.to(device=self.idle_device, dtype=torch.uint8)
        images = torch.squeeze(images).numpy()
        return images
    
    def _encode_context(self, prompt, negative_prompt=None, use_cfg=True):
        positive_tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        positive_tokens = positive_tokens.to(device=self.device, dtype=torch.long)
        # (1, 77, 768)
        context = self.clip(positive_tokens)

        if use_cfg:
            negative_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

            negative_tokens = negative_tokens.to(device=self.device, dtype=torch.long)
            uncond_context = self.clip(negative_tokens)

            # -> (2, 77, 768)
            context = torch.cat((context, uncond_context), dim=0)
        return context

    
    def _denoise_latent(
        self,
        sampler,
        latents: torch.Tensor,
        context: torch.Tensor,
        generator: torch.Generator,
        use_cfg: bool = True,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        for i in range(len(sampler.timesteps)):
            # get current and prev timesteps
            timestep = sampler.timesteps[i]
            if i < len(sampler.timesteps) - 1:
                prev_timestep = sampler.timesteps[i + 1]
            else:
                prev_timestep = None # or -1?

            if isinstance(sampler, Euler):
                latent_input=sampler.scale_model_input(
                    latents=latents,
                    timestep=timestep
                )
            else: 
                latent_input = latents # for DDPM and DDIM

            # UNet time input
            time_input = timestep.to(device=self.device).reshape(1) # .reshape(1) ??
            time_input = time_input.expand(latents.shape[0])
            if use_cfg:
                latent_input = torch.cat((latent_input, latent_input), dim=0)
                time_input = torch.cat((time_input, time_input), dim=0)
            context_input = context

            # UNet inference
            noise_pred = self.unet(latent_input, context_input, time_input)

            if use_cfg:
                noise_pred_cond, noise_pred_uncond = torch.chunk(
                    noise_pred, chunks=2, dim=0
                )

                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

            # remove noise
            latents = sampler.step(
                noise_pred=noise_pred, # (B, 4, 64, 64)
                timestep=timestep, # (B,)
                prev_timestep=prev_timestep,
                latents=latents,
                generator=generator
            )
        return latents


    # -----------------------------------------------------------------------------
    # Main generation methods
    # -----------------------------------------------------------------------------

    @torch.no_grad()
    def txt_2_img(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        n_step_inf: int = 50,
        seed: int = 42,
        sampler_name: str = "ddpm"
    ) -> np.ndarray:

        generator = self._make_generator(seed=seed, device=self.device)
        use_cfg = guidance_scale != 1.0

        if not negative_prompt and use_cfg:
            negative_prompt = ""

        # Encode context
        self._load_model(self.clip)
        context = self._encode_context(
            prompt=prompt, negative_prompt=negative_prompt, use_cfg=use_cfg
        )
        self._offload_model(self.clip)

        # Denoise latent
        latents = torch.randn((1, 4, 64, 64), generator=generator, device=self.device)
        sampler = self._make_sampler(sampler_name, n_step_inf)
        sampler.set_timesteps(1.0)
        if isinstance(sampler, Euler):
            latents *= sampler.init_noise_scale.to(device=latents.device, dtype=latents.dtype)

        self._load_model(self.unet)
        latents = self._denoise_latent(
            sampler=sampler,
            latents=latents,
            generator=generator,
            context=context,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
        )
        self._offload_model(self.unet)

        # Decode denoised latent
        self._load_model(self.vae)
        image = self.vae.decode(latents)
        self._offload_model(self.vae)

        return self._postprocess_image(image)

    @torch.no_grad()
    def img_2_img(
        self,
        input_image: np.ndarray,
        strength: float = 0.8,
        prompt: str = "",
        negative_prompt: str = "",
        guidance_scale: float = 7.5,
        n_step_inf: int = 50,
        seed: int = 42,
        sampler_name: str = "ddpm"
    ) -> np.ndarray:

        if input_image is None:
            raise ValueError("Input image required for img2img")
        if not isinstance(input_image, np.ndarray):
            raise TypeError("Input image must be a numpy array")
        if input_image.shape != (512, 512, 3):
            raise ValueError("Input image must have shape (512, 512, 3)")
        if input_image.min() < 0 or input_image.max() > 255:
            raise ValueError("Pixel values must be between 0 and 255")

        if not 0 <= strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        generator = self._make_generator(seed=seed, device=self.device)

        use_cfg = guidance_scale != 1.0

        if not negative_prompt and use_cfg:
            negative_prompt = ""

        # Encode context and input image
        self._load_model(self.clip)
        context = self._encode_context(
            prompt=prompt, negative_prompt=negative_prompt, use_cfg=use_cfg
        )
        self._offload_model(self.clip)

        input_image = torch.tensor(input_image, dtype=torch.float32, device=self.device)
        input_image = torch.unsqueeze(input_image, dim=0)
        input_image = torch.permute(input_image, (0, 3, 1, 2))
        input_image = self._rescale(input_image, 0, 255, -1, 1)

        self._load_model(self.vae)
        encoder_noise = torch.randn(
            (1, 4, 64, 64), generator=generator, device=self.device
        )
        latents = self.vae.encode(input_image, encoder_noise)
        self._offload_model(self.vae)

        # Denoise latent
        sampler = self._make_sampler(sampler_name, n_step_inf)
        sampler.set_timesteps(strength)

        if len(sampler.timesteps) > 0:
            noise = torch.randn(latents.shape, generator=generator, device=latents.device)
            latents = sampler.add_noise(
                latents=latents,
                noise=noise,
                timesteps=sampler.timesteps[0]
            )

        self._load_model(self.unet)
        latents = self._denoise_latent(
            sampler=sampler,
            latents=latents,
            context=context,
            generator=generator,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
        )
        self._offload_model(self.unet)

        # Decode denoised latent
        self._load_model(self.vae)
        image = self.vae.decode(latents)
        self._offload_model(self.vae)

        return self._postprocess_image(image)
