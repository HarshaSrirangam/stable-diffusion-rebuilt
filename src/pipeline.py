import numpy as np
import torch
from tqdm import tqdm
from ddpm import DDPMSampler

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

HEIGHT = 512
WIDTH = 512
LATENT_WIDTH = 512 // 8
LATENT_HEIGHT = 512 // 8

@torch.no_grad()
def generate(
    prompt: str,
    negative_prompt: str,
    input_image=None,
    strength=0.8, do_cfg=True,
    cfg_scale=7.5,
    sampler_name='ddpm',
    n_step=50, 
    models={}, 
    seed=None,
    device='None',
    idle_device='None',
    tokenizer=None
):
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1!!") 
    if idle_device:
        to_idle: lambda x: x.to(device)
    else: to_idle: lambda x: x

    # seed generator for noise
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    # CLIP
    clip = models['clip']
    clip.to(device)

    # cfg
    if do_cfg:
        # tokenize prompt
        pos_tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
        neg_tokens = tokenizer.batch_encode_plus([negative_prompt], padding='max_length', max_length=77)
        # (1, 77)
        pos_tokens = torch.Tensor(pos_tokens, dtype=torch.long, device=device)
        neg_tokens = torch.Tensor(neg_tokens, dtype=torch.long, device=device)
        # (1, 77) -> (B, seq_len, 768)
        pos_context = clip(pos_tokens)
        neg_context = clip(neg_tokens)

        # (2, 77, 768)
        context = torch.cat((pos_context, neg_context), dim=0)
    else:
        tokens = tokenizer.batch_encode_plus([prompt], padding='max_length', max_length=77).input_ids
        tokens = torch.Tensor(tokens, dtype=torch.long, device=device)
        # (1, 77, 768)
        context = clip(tokens)
    
    # move clip back to cpu
    to_idle(clip)

    # load sampler
    if sampler_name == 'ddpm':
        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(n_step)
    else:
        raise ValueError('invalid sampler!!')
    
    latent_shape = (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
    
    if input_image:
        # image to image
        encoder = models['encoder']
        encoder.to(device)

        # resize to 512x512
        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        input_image_tensor = torch.Tensor(input_image_tensor, dtype=torch.float32)

        # rescale and fix shape
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
        # (H, W, C) -> (1, H, W, C) -> (1, C, H, W)
        input_image_tensor = input_image_tensor.unsqueeze(0).permute(0, 3, 1, 2)

        # encode
        encoder_noise = torch.randn(latent_shape, generator=generator, device=device)
        latent = encoder(input_image_tensor, encoder_noise)

        sampler.set_strength(strength=strength)
        latent = sampler.add_noise(latent, sampler.timesteps[0])
        to_idle(encoder)
    else:
        # text to image
        latent = torch.randn(latent_shape, generator=generator, device=device)


    # diffusion
    diffusion = models['diffusion']
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)
    for i, timestep in enumerate(timesteps):
        time_embedding = get_time_embedding(timestep).to(device)

        # (1, 4, LATENT_HEIGHT, LATENT_WIDTH)
        model_input = latent
        
        if do_cfg:
            # repeat along batch dim: (2, 4, LATENT_HEIGHT, LATENT_WIDTH)
            model_input = torch.cat((model_input, model_input), dim=0)

        model_output = diffusion(model_input, context, time_embedding)
        output_pos, output_neg = model_output.chunk(chunks=2, dim=0)
        model_output = cfg_scale * (output_pos - output_neg) + output_neg

        # remove noise
        latent = sampler.step(timestep, latent, model_output)

    to_idle(diffusion)
    
    # decode
    decoder = models['decoder']
    decoder.to(device)

    image = decoder(latent)
    to_idle(decoder)
    image = rescale(image, (-1, 1), (0, 255), clamp=True)
    # (1, C, H, W) -> (1, H, W, C) -> (H, W, C)
    image = image.permute(0, 2, 1, 3).squeeze()
    image = image.to('cpu', torch.uint8).numpy()

    return image

def rescale(x: torch.Tensor, old_range, new_range, clamp=False) -> torch.Tensor:
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min 
    if clamp:
        x = x.clamp(new_min, new_max)

    return x 

def get_time_embedding(timestep):
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)

