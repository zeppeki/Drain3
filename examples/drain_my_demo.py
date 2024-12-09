import logging
import sys
import os
import subprocess
from os.path import dirname
import pandas as pd
import time
from collections import defaultdict


from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from drain3.masking import MaskingInstruction
from drain3.masking import LogMasker

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")


def download_dataset(dataset):
    DATASET_BASE_URL = f"https://raw.githubusercontent.com/logpai/loghub-2.0/refs/heads/main/2k_dataset/"
    FILEPATH = f"{dataset}/{dataset}_2k.log_structured.csv"
    in_log_file = f"2k_dataset/{dataset}_2k.log_structured.csv"
    if not os.path.isfile(in_log_file):
        logger.info(f"Downloading file {in_log_file}")
        p = subprocess.Popen(
            f"curl {DATASET_BASE_URL}{FILEPATH} --output {in_log_file}",
            shell=True,
        )
        p.wait()

    return in_log_file


def init_template_miner():
    config = TemplateMinerConfig()
    config.load(f"{dirname(__file__)}/drain3.ini")
    config.profiling_enabled = True
    return TemplateMiner(config=config)


def learning(template_miner, lines):
    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    line_count = 0
    for line in lines:
        result = template_miner.add_log_message(line)
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            logger.info(
                f"Processing line: {line_count}, rate {rate:.1f} lines/sec, "
                f"{len(template_miner.drain.clusters)} clusters so far."
            )
            batch_start_time = time.time()
        if result["change_type"] != "none":
            logger.info(f"Input ({line_count}): {line}")
            logger.info(f"Result: {result}")

    time_took = time.time() - start_time
    rate = line_count / time_took
    logger.info(
        f"--- Done processing file in {time_took:.2f} sec. Total of {line_count} lines, rate {rate:.1f} lines/sec, "
        f"{len(template_miner.drain.clusters)} clusters"
    )

    sorted_clusters = sorted(
        template_miner.drain.clusters, key=lambda it: it.size, reverse=True
    )
    for cluster in sorted_clusters:
        logger.info(cluster)

    print("Prefix Tree:")
    template_miner.drain.print_tree()
    template_miner.profiler.report(0)


datasets = [
    "Proxifier",
    "Linux",
    "Apache",
    "Zookeeper",
    "Mac",
    "OpenStack",
    "HealthApp",
    "Hadoop",
    "HPC",
    "OpenSSH",
    "BGL",
    "HDFS",
    "Spark",
    "Thunderbird",
]

# datasets = ["BGL"]


def calculate_accuracy(ground_truth, predicted_template):
    correct = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == predicted_template[i]:
            correct += 1

    accuracy = correct / len(ground_truth)
    logger.info(f"Accuracy: {accuracy:.2f}")

    return accuracy


def predict(template_miner, log_line):
    cluster = template_miner.match(log_line)
    if cluster is None:
        return "None"
    template = cluster.get_template()
    logger.debug(f"Matched template #{cluster.cluster_id}: {template}")
    logger.debug(f"Parameters: {template_miner.get_parameter_list(template, log_line)}")
    return template


benchmark_settings = {
    "BGL": {
        "regex": [
            {
                "regex_pattern": "core\\.\\d+",
                "mask_with": "CORE",
            }
        ],
        "st": 0.6,
        "depth": 4,
    },
    "HDFS": {
        "regex": [
            {
                "regex_pattern": "blk_-?\\d+",
                "mask_with": "BLK-ID",
            },
            {
                "regex_pattern": "(/|)(\\d+\\.){3}\\d+(:\\d+|)",
                "mask_with": "IP-ADDR",
            },
        ],
        "st": 0.5,
        "depth": 4,
    },
    "HPC": {
        "regex": [
            {
                "regex_pattern": "=\\d+",
                "mask_with": "EQNUM",
            }
        ],
        "st": 0.5,
        "depth": 4,
    },
    "OpenSSH": {
        "regex": [
            {
                "regex_pattern": "(\\d+\.){3}\\d+",
                "mask_with": "IP-ADDR",
            },
            {
                "regex_pattern": "([\\w-]+\\.){2,}[\\w-]+",
                "mask_with": "HOSTNAME",
            },
        ],
        "st": 0.6,
        "depth": 5,
    },
    "OpenStack": {
        "regex": [
            {
                "regex_pattern": "\\[instance:\\s*(.*?)\\]",
                "mask_with": "INSTANCE",
            },
            {
                "regex_pattern": "((\\d+\\.){3}\\d+,?)+",
                "mask_with": "IP-ADDR",
            },
            {
                "regex_pattern": "/.+?\\s",
                "mask_with": "PATH",
            },
            {
                "regex_pattern": "\\d+",
                "mask_with": "NUM",
            },
        ],
        "st": 0.2,
        "depth": 4,
    },
    "Proxifier": {
        "regex": [
            {
                "regex_pattern": "<\\d+\\ssec",
                "mask_with": "NUM-sec",
            },
            {
                "regex_pattern": "([\\w-]+\\.)+[\\w-]+(:\\d+)?",
                "mask_with": "HOSTNAME",
            },
            {
                "regex_pattern": "\\d{2}:\\d{2}(:\\d{2})*",
                "mask_with": "TIME",
            },
            {
                "regex_pattern": "[KGTM]B",
                "mask_with": "SIZE",
            },
        ],
        "st": 0.6,
        "depth": 3,
    },
    "Spark": {
        "regex": [
            {
                "regex_pattern": "(\\d+\.){3}\\d+",
                "mask_with": "IP-ADDR",
            },
            {
                "regex_pattern": "\\b[KGTM]?B\\b",
                "mask_with": "SIZE",
            },
            {
                "regex_pattern": "([\\w-]+\\.){2,}[\\w-]+",
                "mask_with": "HOSTNAME",
            },
        ],
        "st": 0.5,
        "depth": 4,
    },
    "Thunderbird": {
        "regex": [
            {
                "regex_pattern": "(\\d+\\.){3}\\d+",
                "mask_with": "IP-ADDR",
            }
        ],
        "st": 0.5,
        "depth": 4,
    },
    "Zookeeper": {
        "regex": [
            {
                "regex_pattern": "'(/|)(\\d+\.){3}\\d+(:\\d+)?'",
                "mask_with": "IP-ADDR",
            }
        ],
        "st": 0.5,
        "depth": 4,
    },
    "Apache": {
        "regex": [
            {
                "regex_pattern": "(\\d+\.){3}\\d+",
                "mask_with": "IP-ADDR",
            }
        ],
        "st": 0.5,
        "depth": 4,
    },
    "HealthApp": {
        "regex": [
            {
                "regex_pattern": "\\d+##\\d+##\\d+##\\d+##\\d+##\\d+",
                "mask_with": "NUM",
            },
            {
                "regex_pattern": "=\\d+",
                "mask_with": "EQNUM",
            },
        ],
        "st": 0.2,
        "depth": 4,
    },
    "Hadoop": {
        "regex": [
            {
                "regex_pattern": "(\\d+\.){3}\\d+",
                "mask_with": "IP-ADDR",
            },
        ],
        "st": 0.5,
        "depth": 4,
    },
    "Linux": {
        "regex": [
            {
                "regex_pattern": "(\\d+\.){3}\\d+",
                "mask_with": "IP-ADDR",
            },
            {
                "regex_pattern": "\\d{2}:\\d{2}:\\d{2}",
                "mask_with": "TIME",
            },
        ],
        "st": 0.39,
        "depth": 6,
    },
    "Mac": {
        "regex": [
            {
                "regex_pattern": "([\\w-]+\\.){2,}[\\w-]+",
                "mask_with": "HOSTNAME",
            },
        ],
        "st": 0.7,
        "depth": 6,
    },
    "Windows": {
        "regex": [
            {
                "regex_pattern": "0x.*?\\s",
                "mask_with": "HEX",
            },
        ],
        "st": 0.7,
        "depth": 5,
    },
    "Android": {
        "regex": [
            {
                "regex_pattern": "(/[\\w-]+)+",
                "mask_with": "PATH",
            },
            {
                "regex_pattern": "([\\w-]+\\.){2,}[\\w-]+",
                "mask_with": "HOSTNAME",
            },
            {
                "regex_pattern": "\\b(\\-?\\+?\\d+)\\b|\\b0[Xx][a-fA-F\d]+\\b|\\b[a-fA-F\\d]{4,}\\b",
                "mask_with": "NUM",
            },
        ],
        "st": 0.2,
        "depth": 6,
    },
}
mask_prefix = "<:"
mask_suffix = ":>"


def gen_masking_instructions(masking_list):
    masking_instructions = []
    for mi in masking_list:
        instruction = MaskingInstruction(mi["regex_pattern"], mi["mask_with"])
        masking_instructions.append(instruction)
    return masking_instructions


for dataset in datasets:
    logger.info(f"Processing Start: {dataset}")

    in_log_file = download_dataset(dataset)
    df = pd.read_csv(in_log_file)
    lines = df["Content"].tolist()
    template_miner = init_template_miner()

    # マスキング設定の上書き
    if dataset in benchmark_settings:
        print("replacing masker")
        masking_list = benchmark_settings[dataset]["regex"]
        masking_instructions = gen_masking_instructions(masking_list)
        template_miner.masker = LogMasker(
            masking_instructions, mask_prefix, mask_suffix
        )

        st = benchmark_settings[dataset].get("st")
        if st:
            template_miner.drain.sim_th = st
        depth = benchmark_settings[dataset].get("depth")
        if depth:
            template_miner.drain.log_cluster_depth = depth
            template_miner.drain.max_node_depth = depth - 2

    # 学習
    learning(template_miner, lines)
    logger.info(f"Done: {dataset}")

    # 予測
    predicted_template = [predict(template_miner, line) for line in lines]
    ground_truth = df["EventTemplate"].tolist()

    accuracy = calculate_accuracy(ground_truth, predicted_template)
    logger.info(f"Done: {dataset}, accuracy: {accuracy:.2f}")

    # パラメータの抽出
    param_dict = defaultdict(set)
    for line in lines:
        cluster = template_miner.match(line)
        if cluster is None:
            continue
        cluster_id = cluster.cluster_id
        template = cluster.get_template()
        params = template_miner.extract_parameters(template, line)

        if not params:
            continue

        for idx, param in enumerate(params):
            param_dict[(cluster_id, idx)].add(param)

    for key, value in param_dict.items():
        logger.info(f"{key=}")
        for v in value:
            logger.info(f"  {v}")
