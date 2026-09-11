from alcatraz.clusters.local import DEFAULT_LIMITS, LocalConfig


def test_local_config_forwards_container_memory_limits() -> None:
    cluster = LocalConfig(
        image="example:test",
        shm_size="8g",
        mem_limit="32g",
    ).build()

    assert cluster.shm_size == "8g"
    assert cluster.mem_limit == "32g"


def test_docker_client_timeout_exceeds_paperbench_reproduction_timeout() -> None:
    assert DEFAULT_LIMITS["docker_client_timeout_seconds"] == 25 * 60 * 60
