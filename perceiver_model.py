#!/usr/bin/env python3
import torch
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverImagePreprocessor


class PostProcessor(torch.nn.Module):
    def __init__(self, image_channels, image_height, image_width):
        super().__init__()
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

    def forward(self, x, *args, **keywords):
        x = x.permute([0, 2, 1])
        x = x.view([-1, self.image_channels, self.image_height, self.image_width])
        x = torch.sigmoid(x)
        return x


class PerceiverImageReconstructModel(PerceiverModel):
    def __init__(self, image_channels, image_height, image_width):
        hidden_size = 32
        config = PerceiverConfig(d_model=hidden_size, d_latents=80)
        super().__init__(config)
        self.config = config
        self.input_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",
            spatial_downsample=1,
            in_channels=image_channels,
            out_channels=hidden_size,
            position_encoding_type="trainable",
            concat_or_add_pos="add",
            project_pos_dim=hidden_size,
            trainable_position_encoding_kwargs=dict(
                index_dims=image_height * image_width, num_channels=hidden_size
            )
        )
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=image_channels,
            num_channels=hidden_size * 2,
            concat_preprocessed_input=True,
            trainable_position_encoding_kwargs=dict(
                index_dims=image_height * image_width, num_channels=hidden_size
            ))
        self.output_postprocessor = PostProcessor(image_channels, image_height, image_width)


class PerceiverRapperModel(nn.Module):
    def __init__(self, image_channels, image_height, image_width):
        super(PerceiverRapperModel, self).__init__()
        self.model = PerceiverImageReconstructModel(image_channels, image_height, image_width)

    def forward(self, x):
        out = self.model(x)
        return out.logits


if __name__ == "__main__":
    from constant import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL
    BATCH_SIZE = 2
    model = PerceiverRapperModel(IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH).cuda()
    x = torch.ones([BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH]).cuda()
    out = model(x)
    print(out)
