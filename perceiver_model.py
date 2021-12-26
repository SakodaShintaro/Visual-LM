#!/usr/bin/env python3
import torch
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder, PerceiverImagePreprocessor
from constant import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL


class PostProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **keywords):
        x = x.permute([0, 2, 1])
        x = x.view([-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH])
        x = torch.sigmoid(x)
        return x


class PerceiverSegmentationModel(PerceiverModel):
    def __init__(self, input_num_channels):
        hidden_size = 32
        config = PerceiverConfig(d_model=hidden_size, d_latents=80)
        super().__init__(config)
        self.config = config
        self.input_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",
            spatial_downsample=1,
            in_channels=input_num_channels,
            out_channels=hidden_size,
            position_encoding_type="trainable",
            concat_or_add_pos="add",
            project_pos_dim=hidden_size,
            trainable_position_encoding_kwargs=dict(
                index_dims=IMAGE_HEIGHT * IMAGE_WIDTH, num_channels=hidden_size
            )
        )
        self.decoder = PerceiverBasicDecoder(
            config,
            output_num_channels=input_num_channels,
            num_channels=hidden_size * 2,
            concat_preprocessed_input=True,
            trainable_position_encoding_kwargs=dict(
                index_dims=IMAGE_HEIGHT * IMAGE_WIDTH, num_channels=hidden_size
            ))
        self.output_postprocessor = PostProcessor()

    def postprocess(logits: torch.Tensor, *args, **keywords):
        return logits


class PerceiverSegModel(nn.Module):
    def __init__(self, input_channel_num):
        super(PerceiverSegModel, self).__init__()
        self.main_model = PerceiverSegmentationModel(input_num_channels=input_channel_num)

    def forward(self, x):
        out = self.main_model(x)
        return out.logits


if __name__ == "__main__":
    BATCH_SIZE = 2
    model = PerceiverSegModel(input_channel_num=IMAGE_CHANNEL).cuda()
    x = torch.ones([BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH]).cuda()
    out = model(x)
    print(out)
