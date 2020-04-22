"""FSL based interfaces that break when using testing."""
from nipype.interfaces.fsl.maths import MathsInput, MathsCommand
from nipype.interfaces.base import File


class _ApplyMaskInput(MathsInput):

    in_file = File(
        position=2, argstr="%s", mandatory=True,
        desc="Image on which to perform operations")

    mask_file = File(
        mandatory=True,
        argstr="-mas %s",
        position=4,
        desc="Image defining mask spaces")


class ApplyMask(MathsCommand):
    """Use fslmaths to apply a binary mask to another image."""

    input_spec = _ApplyMaskInput
    _suffix = "_masked"
