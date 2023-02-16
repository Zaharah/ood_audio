import tensorflow as tf
import tensorflow_addons as tfa

class MelFilterbanks(tf.keras.layers.Layer):
  def __init__(self,
      n_filters = 64,
      sample_rate = 16000,
      n_fft = 512,
      window_len = 25.,
      window_stride = 10.,
      min_freq = 60.0,
      max_freq = 7800.0,
      **kwargs):
    super().__init__(**kwargs)

    self._n_filters = n_filters
    self._sample_rate = sample_rate
    self._n_fft = n_fft
    self._window_len = int(sample_rate * window_len // 1000 + 1)
    self._window_stride = int(sample_rate * window_stride // 1000)
    self._min_freq = min_freq
    self._max_freq = max_freq if max_freq else sample_rate / 2.

    self.mel_filters = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=self._n_filters,
        num_spectrogram_bins=self._n_fft // 2 + 1,
        sample_rate=self._sample_rate,
        lower_edge_hertz=self._min_freq,
        upper_edge_hertz=self._max_freq)

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    if inputs.shape.ndims == 3:
      if inputs.shape[-1] != 1:
        raise ValueError("Only one channel supported but got inputs"
          f" with shape {inputs.shape}")
      inputs = tf.squeeze(inputs, axis=-1)

    stft = tf.signal.stft(
        inputs,
        frame_length=self._window_len,
        frame_step=self._window_stride,
        fft_length=self._n_fft,
        pad_end=True)

    spectrogram = tf.math.square(tf.math.abs(stft))
    mel_filterbanks = tf.matmul(spectrogram, self.mel_filters)
    mel_filterbanks = tf.math.log(mel_filterbanks + 1e-5)
    return mel_filterbanks
