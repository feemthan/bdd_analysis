import toml

def get_pip_requirements_from_pyproject(
    pyproject_path: str = "pyproject.toml",
    project_key: str = "project",
    dependencies_key: str = "dependencies",
    uv_tool_key: str = "uv",
    uv_index_key: str = "index",
    uv_sources_key: str = "sources",
) -> list[str]:
    data = toml.load(pyproject_path)

    # 1. Base dependencies
    deps = data.get(project_key, {}).get(dependencies_key, [])
    if not isinstance(deps, list):
        raise ValueError(f"{project_key}.{dependencies_key} must be a list")

    # 2. Build map of named UV indices → URLs
    uv = data.get("tool", {}).get(uv_tool_key, {})
    indices = uv.get(uv_index_key, [])
    index_map = {idx["name"]: idx["url"] for idx in indices if "name" in idx and "url" in idx}

    # 3. Collect URLs from UV sources
    extra_urls = set()
    for _, srcs in uv.get(uv_sources_key, {}).items():
        for src in srcs:
            url = index_map.get(src.get("index"))
            if url:
                extra_urls.add(url)

    # 4. Assemble pip_requirements
    pip_requirements = []
    for url in extra_urls:
        pip_requirements.append(f"--extra-index-url {url}")
    pip_requirements.extend([dep for dep in deps if not dep.lower().startswith("python")])

    return pip_requirements


def log_model_with_pyproject_env(
    pyproject_path: str = "pyproject.toml",
) -> list[str]:
    pip_reqs = get_pip_requirements_from_pyproject(pyproject_path)
    return pip_reqs
