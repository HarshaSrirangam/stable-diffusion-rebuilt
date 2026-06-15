import numpy as np
import torch
from transformers import CLIPTokenizer

from sd15.convert_weights import load_all
from sd15.ddpm import DDPM
from sd15.model.autoencoder import Autoencoder
from sd15.model.clip import CLIP
from sd15.model.unet import UNet


class SD15InferencePipeline:
    def __init__(
        self,
        ckpt_path,
        n_step_train: int,
        device: torch.device | str,
        idle_device: torch.device | str,
    ):
        self.device = device
        self.idle_device = idle_device
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vae = Autoencoder()
        self.clip = CLIP()
        self.unet = UNet()

        if n_step_train <= 0:
            raise ValueError("n_step_train must be positive")
        self.n_step_train = n_step_train
        self.sampler = DDPM(n_step_train=n_step_train)

        load_all(ckpt_path, vae=self.vae, clip=self.clip, unet=self.unet)

    def _make_generator(self, seed: int, device: torch.device | str) -> torch.Generator:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator

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
        latent: torch.Tensor,
        inf_timesteps: torch.Tensor,
        context: torch.Tensor,
        generator: torch.Generator,
        use_cfg=True,
        guidance_scale=7.5,
    ) -> torch.Tensor:
        for i in range(len(inf_timesteps)):
            t = int(inf_timesteps[i].item())

            if i < len(inf_timesteps) - 1:
                t_prev = int(inf_timesteps[i + 1].item())
            else:
                t_prev = -1

            time_input = torch.tensor([t], device=self.device)
            latent_input = latent
            if use_cfg:
                latent_input = torch.cat((latent, latent), dim=0)
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
            latent = self.sampler.p_sample(
                latent=latent,
                noise_pred=noise_pred,
                t=t,
                t_prev=t_prev,
                generator=generator,
            )

        return latent

    def _postprocess_image(self, image):
        image = self._rescale(image, -1, 1, 0, 255)
        image = torch.permute(image, (0, 2, 3, 1))
        image = image.to(device=self.idle_device, dtype=torch.uint8)
        image = torch.squeeze(image).numpy()
        return image

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
    ) -> np.ndarray:

        if n_step_inf <= 0:
            raise ValueError("n_step_inf must be greater than 0")
        if self.n_step_train % n_step_inf != 0:
            raise ValueError("n_step_train must be divisible by n_step_inf")

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
        latent = torch.randn((1, 4, 64, 64), generator=generator, device=self.device)

        inf_timesteps = torch.arange(
            start=self.n_step_train - 1, end=-1, step=-(self.n_step_train // n_step_inf)
        )

        self._load_model(self.unet)
        latent = self._denoise_latent(
            inf_timesteps=inf_timesteps,
            latent=latent,
            generator=generator,
            context=context,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
        )
        self._offload_model(self.unet)

        # Decode denoised latent
        self._load_model(self.vae)
        image = self.vae.decode(latent)
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

        if n_step_inf <= 0:
            raise ValueError("n_step_inf must be greater than 0")
        if self.n_step_train % n_step_inf != 0:
            raise ValueError("n_step_train must be divisible by n_step_inf")
        
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
        latent = self.vae.encode(input_image, encoder_noise)
        self._offload_model(self.vae)

        # Denoise latent
        inf_timesteps = torch.arange(
            start=self.n_step_train - 1, end=-1, step=-(self.n_step_train // n_step_inf)
        )

        num_denoising_steps = int(strength * n_step_inf)
        start_idx = n_step_inf - num_denoising_steps
        inf_timesteps = inf_timesteps[start_idx:]

        if len(inf_timesteps) > 0:
            noise_timestep = inf_timesteps[0].unsqueeze(0)
            latent = self.sampler.q_sample(
                latents=latent, timesteps=noise_timestep, generator=generator
            )

        self._load_model(self.unet)
        latent = self._denoise_latent(
            inf_timesteps=inf_timesteps,
            latent=latent,
            generator=generator,
            context=context,
            use_cfg=use_cfg,
            guidance_scale=guidance_scale,
        )
        self._offload_model(self.unet)

        # Decode denoised latent
        self._load_model(self.vae)
        image = self.vae.decode(latent)
        self._offload_model(self.vae)

        return self._postprocess_image(image)
