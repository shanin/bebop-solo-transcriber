import argparse
import shutil
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--destination', type=str, required=True)
    args = parser.parse_args()

    suffix = '_(Woodwinds)_17_HP-Wind_Inst-UVR.wav'
    for file in os.listdir(args.destination):
        if file.endswith(suffix):
            try:
                os.remove(os.path.join(args.source, file.split(suffix)[0] + '.wav'))
            except FileNotFoundError:
                pass