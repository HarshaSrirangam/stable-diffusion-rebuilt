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
        device: torch.device | str = "cuda",
        idle_device: torch.device | str = "cpu",
    ):
        self.device = device
        self.idle_device = idle_device

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.vae = Autoencoder()
        self.clip = CLIP()
        self.unet = UNet()
        self.sampler = DDPM()

        load_all(ckpt_path, vae=self.vae, clip=self.clip, unet=self.unet)

    def _make_generator(self, seed: int, device: torch.device) -> torch.Generator:
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

    @torch.no_grad()
    def txt_2_img(
        self,
        cond_prompt: str,
        uncond_prompt: str = None,
        do_cfg=True,
        cfg_weight=7.5,
        n_step_inf: int = 50,
        seed: int = 42,
    ) -> np.ndarray:

        generator = self._make_generator(seed=seed, device=self.device)

        if not uncond_prompt and do_cfg:
            uncond_prompt = ""

        # encode context
        self._load_model(self.clip)

        cond_tokens = self.tokenizer(
            cond_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        cond_tokens = cond_tokens.to(device=self.device, dtype=torch.long)
        # (1, 77, 768)
        context = self.clip(cond_tokens)

        if do_cfg:
            uncond_tokens = self.tokenizer(
                uncond_prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )["input_ids"]

            uncond_tokens = uncond_tokens.to(device=self.device, dtype=torch.long)
            uncond_context = self.clip(uncond_tokens)

            # -> (2, 77, 768)
            context = torch.cat((context, uncond_context), dim=0)

        self._offload_model(self.clip)

        latent = torch.randn((1, 4, 64, 64), generator=generator, device=self.device)
        self._load_model(self.unet)

        if 1000 % n_step_inf != 0:
            raise ValueError("1000 must be divisible by n_step_inf")

        inf_timesteps = torch.arange(start=999, end=-1, step=-(1000 // n_step_inf))

        # denoising
        for i in range(len(inf_timesteps)):
            t = int(inf_timesteps[i].item())

            if i < len(inf_timesteps) - 1:
                t_prev = int(inf_timesteps[i + 1].item())
            else:
                t_prev = -1

            time_input = torch.tensor([t], device=self.device)
            latent_input = latent
            if do_cfg:
                latent_input = torch.cat((latent, latent), dim=0)
                time_input = torch.cat((time_input, time_input), dim=0)
            context_input = context

            # UNet inference
            noise_pred = self.unet(latent_input, context_input, time_input)

            if do_cfg:
                noise_pred_cond, noise_pred_uncond = torch.chunk(
                    noise_pred, chunks=2, dim=0
                )

                noise_pred = noise_pred_uncond + cfg_weight * (
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

        self._offload_model(self.unet)

        # decode denoised latent
        self._load_model(self.vae)

        image = self.vae.decode(latent)
        self._offload_model(self.vae)
        image = self._rescale(image, -1, 1, 0, 255)
        image = torch.permute(image, (0, 2, 3, 1))
        image = image.to(device=self.idle_device, dtype=torch.uint8)
        image = torch.squeeze(image).numpy()

        return image
