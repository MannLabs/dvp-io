from dvpio._utils import deprecated_docs, deprecated_log

from .shapes.lmd_writer import write_lmd as _write_lmd


# TODO: Remove with dvpio v0.6.0
@deprecated_log(
    "dvpio.write.write_lmd is deprecated. It will be removed in the next minor release. Use the equivalent dvpio.write.shapes.write_lmd instead."
)
@deprecated_docs
def write_lmd(*args, **kwargs):
    """Deprecated wrapper. Use :func:`dvpio.write.shapes.write_lmd` instead."""
    return _write_lmd(*args, **kwargs)
