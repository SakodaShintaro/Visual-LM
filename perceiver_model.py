#!/usr/bin/env python3

import torch
from torch import nn
from transformers import PerceiverModel
from transformers.models.perceiver.configuration_perceiver import PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder
from cnn_model import Conv2dWithBatchNorm
from constant import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL
from transformers.models.perceiver.modeling_perceiver import PerceiverImagePreprocessor

class PostProcessor(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, *args, **keywords):
        x = x.permute([0, 2, 1])
        x = x.view([-1, 1, IMAGE_HEIGHT, IMAGE_WIDTH])
        x = torch.sigmoid(x)
        return x



class PerceiverSegmentationModel(PerceiverModel):
    def __init__(self, config, pos_encoding_num_channels, output_num_channels):
        super().__init__(config)
        self.config = config
        trainable_position_encoding_kwargs = dict(
            index_dims=IMAGE_HEIGHT * IMAGE_WIDTH, num_channels=pos_encoding_num_channels
        )
        self.input_preprocessor = PerceiverImagePreprocessor(
            config,
            prep_type="conv1x1",
            spatial_downsample=1,
            in_channels=1,
            out_channels=pos_encoding_num_channels,
            position_encoding_type="trainable",
            concat_or_add_pos="add",
            project_pos_dim=pos_encoding_num_channels,
            trainable_position_encoding_kwargs=trainable_position_encoding_kwargs
        )
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=output_num_channels, num_channels=pos_encoding_num_channels,
                                             trainable_position_encoding_kwargs=trainable_position_encoding_kwargs)
        self.output_postprocessor = PostProcessor()

    def postprocess(logits: torch.Tensor, *args, **keywords):
        return logits


class PerceiverSegModel(nn.Module):
    def __init__(self, input_channel_num):
        super(PerceiverSegModel, self).__init__()
        hidden_size = 32
        config = PerceiverConfig(d_model=hidden_size, d_latents=80)
        self.main_model = PerceiverSegmentationModel(config, pos_encoding_num_channels=hidden_size, output_num_channels=input_channel_num)

    def forward(self, x):
        input_shape = x.shape
        out = self.main_model(x)
        x = out.logits
        return x


if __name__ == "__main__":
    BATCH_SIZE = 2
    model = PerceiverSegModel(input_channel_num=IMAGE_CHANNEL).cuda()
    x = torch.ones([BATCH_SIZE, IMAGE_CHANNEL, IMAGE_HEIGHT, IMAGE_WIDTH]).cuda()
    out = model(x)
    print(out)
