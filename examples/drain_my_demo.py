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


for dataset in datasets:
    in_log_file = download_dataset(dataset)
    df = pd.read_csv(in_log_file)
    lines = df["Content"].tolist()
    template_miner = init_template_miner()
    learning(template_miner, lines)
    logger.info(f"Done: {dataset}")

    predicted_template = [predict(line) for line in lines]
    ground_truth = df["EventTemplate"].tolist()

    accuracy = calculate_accuracy(ground_truth, predicted_template)
    logger.info(f"Done: {dataset}, accuracy: {accuracy:.2f}")

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
