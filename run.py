import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.PAM_Diago import PAM_Diago
from src.dataset import ExampleDatasetFolder
from pydub import AudioSegment
import os
import shutil
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PAM")
    parser.add_argument('--folder', type=str, default='./raw_audios',help='Folder path to evaluate')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of examples per batch')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--save_results', type=bool, default=False, help='Save PAM evaluation results to json file')
    args = parser.parse_args()

    # initialize PAM
    pam = PAM_Diago(use_cuda=torch.cuda.is_available())
    
    CONVERTED_FOLDER = './audios_converted'
    
    if os.path.isdir(CONVERTED_FOLDER):
        fichiers_to_suppr = os.listdir(CONVERTED_FOLDER)
        # Supprime chaque fichier
        for fichier in fichiers_to_suppr:
            chemin_fichier = os.path.join(CONVERTED_FOLDER, fichier)
            try:
                if os.path.isfile(chemin_fichier) or os.path.islink(chemin_fichier):
                    os.remove(chemin_fichier)
                elif os.path.isdir(chemin_fichier):
                    os.rmdir(chemin_fichier)
            except Exception as e:
                print(f"Erreur lors de la suppression de {chemin_fichier}: {e}")
    else:
        os.mkdir(CONVERTED_FOLDER)
        
    files = os.listdir(args.folder)
    for file in files:
        nom, extension = os.path.splitext(file)
        source = os.path.join(args.folder,file)
        destination = os.path.join(CONVERTED_FOLDER,nom+'.wav')
       
        if extension != 'wav':
            audio = AudioSegment.from_file(source)
            audio.export(out_f=destination,format='wav')
        else:
            shutil.copy(source, destination)     
        
    
    # Create Dataset and Dataloader
    dataset = ExampleDatasetFolder(
        src=CONVERTED_FOLDER,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle = False,
                            num_workers = args.num_workers,
                            pin_memory = False, drop_last=False, collate_fn=dataset.collate)

    # Evaluate and print PAM score
    files_names, collect_pam, collect_pam_segment = [], [], []
    for files, audios, sample_index in tqdm(dataloader):
        files_names += files
        pam_score, pam_segment_score = pam.evaluate(audios, sample_index)
        collect_pam += pam_score
        collect_pam_segment += pam_segment_score
        
    for file, score in zip(files_names, collect_pam):
        print(f"PAM eval :\n\tFile : {file}\n\tScore : {score}")
    if args.save_results:
        results = [{'file':os.path.basename(file),'pam_score':score} for file,score in zip(files_names, collect_pam)]
        with open('results.json', 'w') as f:
            json.dump(results,f)