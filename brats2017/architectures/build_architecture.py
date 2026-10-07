"""
Architecture router for SegFormer3D and CAPF-Extreme.

Optional CAPF variants can be added beside this router when the final public
implementation is standardized.
"""


def build_architecture(config):
    model_name = config.get("model_name", "segformer3d")

    if model_name == "segformer3d":
        from .segformer3d import build_segformer3d_model

        return build_segformer3d_model(config)

    if model_name in {
        "segformer3d_capf",
        "segformer3d_capf_extreme",
        "constructive_primitive_field",
    }:
        try:
            # Current placement: project root.
            from segformer3d_constructive_primitive_field_extreme import (
                build_segformer3d_model,
            )
        except ImportError:
            # Optional future placement: architectures/ directory.
            from .segformer3d_constructive_primitive_field_extreme import (
                build_segformer3d_model,
            )

        return build_segformer3d_model(config)

    raise ValueError(
        f"Unsupported model_name={model_name!r}. "
        "Supported values: 'segformer3d', 'segformer3d_capf_extreme'."
    )

