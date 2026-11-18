"""
Custom exceptions for the DLM Extraction Pipeline.
"""


class DLMPipelineError(Exception):
    """Base exception for all DLM pipeline errors."""
    pass


class ConfigurationError(DLMPipelineError):
    """Raised when pipeline configuration is invalid or missing."""
    pass


class DataIngestionError(DLMPipelineError):
    """Raised when an error occurs during data loading or reading."""
    pass


class TrackerLoadError(DataIngestionError):
    """Raised when tracker workbook cannot be parsed or UPC is missing."""
    pass


class SchemaParsingError(DLMPipelineError):
    """Raised when table schema definitions cannot be parsed."""
    pass


class ProcessingError(DLMPipelineError):
    """Raised when defect calculation or processing fails."""
    pass


class HierarchyBuildError(DLMPipelineError):
    """Raised when building parent/child or LHS/RHS hierarchy fails."""
    pass


class ImageUploadError(DLMPipelineError):
    """Raised when defect image upload to CDN fails."""
    pass


class ValidationError(DLMPipelineError):
    """Raised when validation checks (V1-V9) fail."""
    pass


class ExportError(DLMPipelineError):
    """Raised when table or report export to Excel/CSV fails."""
    pass
