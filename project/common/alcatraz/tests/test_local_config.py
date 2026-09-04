from alcatraz.clusters.local import LocalConfig


def test_local_config_forwards_container_memory_limits() -> None:
    cluster = LocalConfig(
        image="example:test",
        shm_size="8g",
        mem_limit="32g",
    ).build()

    assert cluster.shm_size == "8g"
    assert cluster.mem_limit == "32g"
