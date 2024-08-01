import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .PAM_Diago import PAM_Diago
from .dataset import ExampleDatasetFiles

from pydub import AudioSegment
import os
import json


def convertFiles(files: list[str]):
    """
    Sort audio files from other files. Keep wav files and convert others extension to wav for scoring.
    """
    files_converted = []
    files_names = []
    for file in files:
        nom, extension = os.path.splitext(file)
        print(extension)
        if extension == ".wav":
            files_converted.append(file)
            files_names.append(file)
        elif extension in [
            ".mp3",
            ".flac",
            ".oga",
            ".fla",
            ".aac",
            ".m4a",
            ".m4p",
            ".m4b",
            ".mp4",
            ".3gp",
            ".ogg",
            ".oga",
        ]:
            audio = AudioSegment.from_file(file)
            files_converted.append(audio.export(format="wav"))
            files_names.append(file)

    return files_converted, files_names


def evaluateFiles(
    files: list[str],
    batch_size: int = 10,
    num_workers: int = 0,
    save_result: bool = False,
) -> list[dict[str, any]] | None:
    """
    Compute PAM score of files.
    """
    PAMEvaluator = PAM_Diago(use_cuda=torch.cuda.is_available())

    files_converted, files_names = convertFiles(files)

    dataset = ExampleDatasetFiles(files_converted)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=dataset.collate,
    )

    collect_pam, collect_pam_segment = [], []
    for files, audios, sample_index in tqdm(dataloader):
        pam_score, pam_segment_score = PAMEvaluator.evaluate(audios, sample_index)
        collect_pam += pam_score
        collect_pam_segment += pam_segment_score

    results = [
        {"file": file, "pam_score": score}
        for file, score in zip(files_names, collect_pam)
    ]

    if save_result:
        with open("results.json", "w") as f:
            json.dump(results, f)

    return results


def evaluateFolder(
    folder: str, batch_size: int = 10, num_workers: int = 0, save_result: bool = False
) -> list[dict[str, any]] | None:
    """
    Compute PAM score for files in folder.
    """
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    return evaluateFiles(
        files=files,
        batch_size=batch_size,
        num_workers=num_workers,
        save_result=save_result,
    )
