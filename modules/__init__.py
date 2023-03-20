from .blur import ParallaxICBlurModel, PointwiseConvBlurModel

# Blur models clases
blur_model_dict = {
    'ICB': ParallaxICBlurModel,
    'PWB': PointwiseConvBlurModel
}