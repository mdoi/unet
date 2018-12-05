#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout

class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 1024
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # (1024 x 1024 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        # エンコーダーの作成
        # (512 x 512 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (256 x 256 x 2N)
        filter_count = first_layer_filter_count * 2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (128 x 128 x 4N)
        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (64 x 64 x 8N)
        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (32 x 32 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (16 x 16 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (8 x 8 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (4 x 4 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        # (2 x 2 x 8N)
        enc9 = self._add_encoding_layer(filter_count, enc8)

        # (1 x 1 x 8N)
        enc10 = self._add_encoding_layer(filter_count, enc9)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, enc10)
        dec1 = concatenate([dec1, enc9], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc8], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc7], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc6], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 8N)
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc5], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 8N)
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc4], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x 4N)
        filter_count = first_layer_filter_count * 4
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc3], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x 2N)
        filter_count = first_layer_filter_count * 2
        dec8 = self._add_decoding_layer(filter_count, False, dec7)
        dec8 = concatenate([dec8, enc2], axis=self.CONCATENATE_AXIS)

        # (512 x 512 x N)
        filter_count = first_layer_filter_count
        dec9 = self._add_decoding_layer(filter_count, False, dec8)
        dec9 = concatenate([dec9, enc1], axis=self.CONCATENATE_AXIS)

        # (1024 x 1024 x output_channel_count)
        dec10 = Activation(activation='relu')(dec9)
        dec10 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec10)
        dec10 = Activation(activation='sigmoid')(dec10)

        self.UNET = Model(input=inputs, output=dec10)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET
