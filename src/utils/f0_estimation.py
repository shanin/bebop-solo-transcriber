import torchaudio
from pesto import load_model
import os
import torch 
import numpy as np
import pandas as pd
import argparse
import pyloudnorm as pyln

def loudnorm(x, sr, target_lufs=-23.0):
  peak_normalized_audio = pyln.normalize.peak(x, -1.0)
  meter = pyln.Meter(sr)
  loudness = meter.integrated_loudness(peak_normalized_audio)
  loudness_normalized_audio = pyln.normalize.loudness(peak_normalized_audio, loudness, target_lufs)
  return loudness_normalized_audio


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_list", type=str, required=True)
  parser.add_argument("--output_folder", type=str, required=True)
  args = parser.parse_args()

  os.makedirs(args.output_folder, exist_ok=True)

  data_list = pd.read_csv(args.data_list)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print('Loading model... on device: ', device)
  pesto_model = load_model("mir-1k_g7", step_size=10.).to(device)

  
  with torch.no_grad():
    for _, row in data_list.iterrows():
      idx = row['example_id']
      if os.path.exists(os.path.join(args.output_folder, idx + ".activations.npy")):
        continue

      x, sr = torchaudio.load(row['clean_solo'])
      x = x.mean(dim=0) 
      x = torch.from_numpy(loudnorm(x.numpy().T, sr)).to(device)

      predictions, confidence, amplitude, activations = pesto_model(x, sr)
      predictions = predictions.to('cpu').detach()
      confidence = confidence.to('cpu').detach()
      amplitude = amplitude.to('cpu').detach()
      activations = activations.to('cpu').detach()

      np.save(os.path.join(args.output_folder, idx + ".activations.npy"), activations.numpy())
      np.save(os.path.join(args.output_folder, idx + ".confidence.npy"), confidence.numpy())
      np.save(os.path.join(args.output_folder, idx + ".predictions.npy"), predictions.numpy())
      np.save(os.path.join(args.output_folder, idx + ".amplitude.npy"), amplitude.numpy())
      

      del x, predictions, confidence, amplitude, activations
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()