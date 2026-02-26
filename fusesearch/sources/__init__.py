from fusesearch.sources.base import SourceAdapter


def create_adapter(source: str, source_config: dict | None = None) -> SourceAdapter:
    """Create a source adapter by name.

    Args:
        source: Adapter type ("local_files", "confluence").
        source_config: Source-specific configuration dict.
            - local_files: {"paths": ["/data/docs"]}
            - confluence: {"spaces": ["DEV", "ENG"]}

    Returns:
        Configured SourceAdapter instance.

    Raises:
        ValueError: If the source type is unknown.
    """
    config = source_config or {}

    if source == "local_files":
        from fusesearch.sources.local_files import LocalFilesAdapter

        return LocalFilesAdapter(directories=config.get("paths", []))

    if source == "confluence":
        from fusesearch.sources.confluence import ConfluenceAdapter

        return ConfluenceAdapter(spaces=config.get("spaces"))

    raise ValueError(f"Unknown source type: {source!r}")
