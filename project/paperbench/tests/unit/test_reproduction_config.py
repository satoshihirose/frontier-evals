from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config

from alcatraz.clusters.local import LocalConfig
from paperbench.nano.structs import ReproductionConfig


def test_reproduction_config_uses_eight_gibibytes_of_shared_memory() -> None:
    computer_config = ReproductionConfig().computer_config

    assert computer_config.shm_size == "8g"

    cluster = task_to_alcatraz_config(computer_config, LocalConfig()).build()

    assert cluster.shm_size == "8g"
