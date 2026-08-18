#!/usr/bin/env python3
"""Generate the six reader-facing Colab notebooks for phases 0 through 5."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "playground"
REPO_URL = (
    "https://github.com/nikolabakic/"
    "Vizuelno-prepoznavanje-govora-na-osnovu-pokreta-usana-pomo-u-LipNet-modela.git"
)
UPSTREAM_URL = "https://github.com/VIPL-Audio-Visual-Speech-Understanding/LipNet-PyTorch.git"
UPSTREAM_SHA = "40209e09c49553c00c25c7d41faa3706aea3c625"
CHECKPOINT_NAME = (
    "LipNet_unseen_loss_0.44562849402427673_wer_0.1332580699113564_"
    "cer_0.06796452465503355.pt"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


def notebook(title: str, cells: list) -> nbf.NotebookNode:
    value = nbf.v4.new_notebook(
        cells=[md(f"# {title}")] + cells,
        metadata={
            "accelerator": "GPU",
            "colab": {"name": title, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.x"},
        },
    )
    for index, cell in enumerate(value.cells):
        cell.id = f"cell-{index:02d}"
    return value


def repo_setup_cell() -> str:
    return f"""
    import os
    import subprocess
    from pathlib import Path

    REPO_URL = {REPO_URL!r}
    REPO = Path('/content/lipnet-serbian')
    if not (REPO / 'lipnet').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
    os.chdir(REPO)
    print('Repo:', REPO)
    print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    """


def drive_setup_cell() -> str:
    return """
    from google.colab import drive

    drive.mount('/content/drive')
    DRIVE_ROOT = Path('/content/drive/MyDrive/LipNet')
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print('Drive izlaz:', DRIVE_ROOT)
    """


def phase0() -> nbf.NotebookNode:
    return notebook(
        "Faza 0 — čist restart i pinovanje VIPL upstream-a",
        [
            md("""
            ## Goal

            Proveri tačan VIPL commit, licencu, mapiranje fajlova i izolaciju legacy toka pre
            pokretanja bilo kakvog modela. Ovaj notebook radi na CPU-u i ne obrađuje podatke.
            """),
            md("## Setup\n\nKloniraj aktivni projekat u privremeni Colab disk."),
            code(repo_setup_cell()),
            md("## Steps\n\n### 1. Pinuj i preuzmi tačnu upstream reviziju"),
            code(f"""
            UPSTREAM_URL = {UPSTREAM_URL!r}
            UPSTREAM_SHA = {UPSTREAM_SHA!r}
            UPSTREAM = Path('/content/VIPL-LipNet-PyTorch')

            remote_line = subprocess.check_output(
                ['git', 'ls-remote', UPSTREAM_URL, 'refs/heads/master'], text=True
            ).strip()
            remote_sha = remote_line.split()[0]
            print('Današnji master:', remote_sha)
            if remote_sha != UPSTREAM_SHA:
                print('Napomena: master se pomerio; projekat i dalje koristi pinovani commit.')

            if not (UPSTREAM / '.git').exists():
                subprocess.run(['git', 'clone', UPSTREAM_URL, str(UPSTREAM)], check=True)
            subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', '--detach', UPSTREAM_SHA], check=True)
            checked_out = subprocess.check_output(
                ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True
            ).strip()
            assert checked_out == UPSTREAM_SHA
            print('VIPL pin potvrđen:', checked_out)
            """),
            md("### 2. Proveri inventar, licencu i evidenciju odstupanja"),
            code("""
            inventory = {
                'model.py': REPO / 'lipnet/model.py',
                'dataset.py': REPO / 'lipnet/dataset.py',
                'cvtransforms.py': REPO / 'lipnet/cvtransforms.py',
                'main.py': REPO / 'lipnet/train.py',
                'demo.py': REPO / 'lipnet/demo.py',
                'options.py': REPO / 'lipnet/options.py',
            }
            for upstream_name, local_path in inventory.items():
                assert (UPSTREAM / upstream_name).exists(), upstream_name
                assert local_path.exists(), local_path
                print(f'{upstream_name:16s} -> {local_path.relative_to(REPO)}')

            license_text = (REPO / 'lipnet/LICENSE.vipl').read_text(encoding='utf-8')
            diff_text = (REPO / 'docs/upstream-diff.md').read_text(encoding='utf-8')
            assert 'MIT License' in license_text and UPSTREAM_SHA in license_text
            assert UPSTREAM_SHA in diff_text
            print('Licenca i upstream-diff su prisutni.')
            """),
            md("## Checks\n\n### 3. Dokaži da aktivni kod ne zavisi od legacy manifesta/ROI modula"),
            code("""
            active_files = [
                *sorted((REPO / 'lipnet').glob('*.py')),
                REPO / 'scripts/prepare_ai_speak.py',
                REPO / 'data/splits.py',
            ]
            forbidden = ('from app', 'import app', 'manifest.csv', 'roi.csv', 'vocab.json', 'split.json')
            violations = []
            for path in active_files:
                text = path.read_text(encoding='utf-8')
                for token in forbidden:
                    if token in text:
                        violations.append((str(path.relative_to(REPO)), token))
            assert not violations, violations
            print('PASS: aktivni kod nema legacy import/artefakt zavisnosti.')
            """),
            code("""
            import json

            result = {
                'phase': 0,
                'upstream_url': UPSTREAM_URL,
                'branch': 'master',
                'commit': UPSTREAM_SHA,
                'inventory': {key: str(value.relative_to(REPO)) for key, value in inventory.items()},
                'legacy_dependencies': [],
            }
            result_path = Path('/content/faza0_result.json')
            result_path.write_text(json.dumps(result, indent=2) + '\\n', encoding='utf-8')
            print(result_path.read_text())
            """),
            md("""
            ## Next Steps

            Faza 0 je završena samo ako svi `assert` pozivi prođu. Zatim otvori notebook Faze 1
            i uključi T4 GPU; ne prelazi na AI-SPEAK pre GRID parity provere.
            """),
        ],
    )


def phase1() -> nbf.NotebookNode:
    return notebook(
        "Faza 1 — originalni VIPL GRID inference i parity",
        [
            md("""
            ## Goal

            Na istom GRID primeru i istom unseen-speaker checkpoint-u poredi originalni
            `model.py` iz pinovanog VIPL commit-a sa minimalno modernizovanim lokalnim modelom.
            Kriterijum je: strict load svih težina, isti logits/decode i upstream CER/WER.
            """),
            md("""
            ## Setup

            Izaberi **Runtime → Change runtime type → T4 GPU**. Landmark detekcija i dva
            model forward-a namerno se izvršavaju na GPU-u.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'face-alignment==1.4.1', 'editdistance>=0.8.1'],
                check=True,
            )
            """),
            code(repo_setup_cell()),
            code(f"""
            import torch

            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            DEVICE = torch.device('cuda')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            print('PyTorch:', torch.__version__)
            print('GPU:', torch.cuda.get_device_name(0))

            UPSTREAM_URL = {UPSTREAM_URL!r}
            UPSTREAM_SHA = {UPSTREAM_SHA!r}
            UPSTREAM = Path('/content/VIPL-LipNet-PyTorch')
            if not (UPSTREAM / '.git').exists():
                subprocess.run(['git', 'clone', UPSTREAM_URL, str(UPSTREAM)], check=True)
            subprocess.run(['git', '-C', str(UPSTREAM), 'checkout', '--detach', UPSTREAM_SHA], check=True)
            assert subprocess.check_output(
                ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True
            ).strip() == UPSTREAM_SHA
            """),
            md("## Steps\n\n### 1. Preuzmi legalni GRID primer, anotaciju i pinovani checkpoint"),
            code(f"""
            from urllib.request import urlretrieve

            WORK = Path('/content/faza1')
            WORK.mkdir(exist_ok=True)
            VIDEO = WORK / 'swwp2s.mpg'
            ALIGN = WORK / 'swwp2s.align'
            CHECKPOINT = WORK / {CHECKPOINT_NAME!r}
            urls = {{
                VIDEO: 'https://spandh.dcs.shef.ac.uk/gridcorpus/examples/id2_vcd_swwp2s.mpg',
                ALIGN: 'https://spandh.dcs.shef.ac.uk/gridcorpus/examples/swwp2s.align',
                CHECKPOINT: (
                    'https://raw.githubusercontent.com/VIPL-Audio-Visual-Speech-Understanding/'
                    f'LipNet-PyTorch/{{UPSTREAM_SHA}}/pretrain/{CHECKPOINT_NAME}'
                ),
            }}
            for path, url in urls.items():
                if not path.exists() or path.stat().st_size == 0:
                    urlretrieve(url, path)
                print(path.name, f'{{path.stat().st_size / 1024**2:.2f}} MiB')
            """),
            md("### 2. Primeni lokalni kod koji čuva VIPL demo geometriju"),
            code("""
            import cv2
            import matplotlib.pyplot as plt
            import numpy as np

            from lipnet.demo import (
                make_face_aligner, mouth_tensor, preprocess_video, write_mouth_jpegs,
            )

            aligner = make_face_aligner(device='cuda', face_detector='sfd')
            processed = preprocess_video(VIDEO, aligner, progress_every=15)
            video = mouth_tensor(processed.frames).unsqueeze(0)
            assert video.shape[1] == 3 and video.shape[-2:] == (64, 128)
            assert 0.0 <= float(video.min()) <= float(video.max()) <= 1.0
            MOUTH_DIR = WORK / 'mouth_jpegs'
            write_mouth_jpegs(processed.frames, MOUTH_DIR)
            print('Ulaz (B,C,T,H,W):', tuple(video.shape))
            print('Landmark frejmovi:', processed.landmark_frames, '/', processed.decoded_frames)

            indices = np.linspace(0, len(processed.frames) - 1, 6, dtype=int)
            fig, axes = plt.subplots(2, 3, figsize=(12, 4))
            for axis, index in zip(axes.flat, indices):
                axis.imshow(cv2.cvtColor(processed.frames[index], cv2.COLOR_BGR2RGB))
                axis.set_title(f'frame {index}')
                axis.axis('off')
            plt.tight_layout()
            plt.show()
            """),
            md("### 3. Učitaj originalni i lokalni model sa svim checkpoint težinama"),
            code("""
            import importlib.util

            from lipnet.model import LipNet as LocalLipNet
            from lipnet.dataset import MyDataset as LocalDataset
            from lipnet.train import load_checkpoint_strict

            spec = importlib.util.spec_from_file_location('vipl_original_model', UPSTREAM / 'model.py')
            upstream_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream_module)

            sys.path.insert(0, str(UPSTREAM))
            dataset_spec = importlib.util.spec_from_file_location(
                'vipl_original_dataset', UPSTREAM / 'dataset.py'
            )
            upstream_dataset_module = importlib.util.module_from_spec(dataset_spec)
            dataset_spec.loader.exec_module(upstream_dataset_module)

            upstream_dataset = upstream_dataset_module.MyDataset.__new__(
                upstream_dataset_module.MyDataset
            )
            upstream_frames = upstream_dataset._load_vid(str(MOUTH_DIR))
            local_frames = LocalDataset._load_vid(MOUTH_DIR)
            np.testing.assert_array_equal(local_frames, upstream_frames)
            video = torch.from_numpy(
                np.ascontiguousarray((local_frames / 255.0).transpose(3, 0, 1, 2))
            ).float().unsqueeze(0)
            print('Dataset _load_vid parity:', local_frames.shape)

            original_model = upstream_module.LipNet().to(DEVICE)
            local_model = LocalLipNet(num_classes=28).to(DEVICE)
            original_audit = load_checkpoint_strict(original_model, CHECKPOINT)
            local_audit = load_checkpoint_strict(local_model, CHECKPOINT)
            assert len(original_audit.loaded) == len(local_audit.loaded)
            print('Strict load:', len(local_audit.loaded), 'tenzora; missing=[]; unexpected=[]')
            """),
            md("## Checks\n\n### 4. Dokaži parity logits-a, dekodiranog teksta, CER-a i WER-a"),
            code("""
            from lipnet.dataset import MyDataset

            original_model.eval()
            local_model.eval()
            with torch.inference_mode():
                original_logits = original_model(video.to(DEVICE))
                local_logits = local_model(video.to(DEVICE))
            torch.testing.assert_close(local_logits, original_logits, rtol=1e-6, atol=1e-6)

            original_prediction = upstream_dataset_module.MyDataset.ctc_arr2txt(
                original_logits.argmax(-1)[0], start=1
            )
            local_prediction = MyDataset.ctc_arr2txt(local_logits.argmax(-1)[0], start=1)
            assert original_prediction == local_prediction

            tokens = []
            for line in ALIGN.read_text().splitlines():
                token = line.split()[-1]
                if token.upper() not in {'SIL', 'SP'}:
                    tokens.append(token.upper())
            truth = ' '.join(tokens)
            original_cer = upstream_dataset_module.MyDataset.cer([original_prediction], [truth])[0]
            original_wer = upstream_dataset_module.MyDataset.wer([original_prediction], [truth])[0]
            cer = MyDataset.cer([local_prediction], [truth])[0]
            wer = MyDataset.wer([local_prediction], [truth])[0]
            assert cer == original_cer and wer == original_wer
            print('Ground truth:', truth)
            print('Original   :', original_prediction)
            print('Lokalni    :', local_prediction)
            print(f'CER={cer:.4f} WER={wer:.4f}')
            """),
            code("""
            import json
            result = {
                'phase': 1,
                'upstream_commit': UPSTREAM_SHA,
                'checkpoint_tensors': len(local_audit.loaded),
                'input_shape': list(video.shape),
                'decoded_frames': processed.decoded_frames,
                'landmark_frames': processed.landmark_frames,
                'truth': truth,
                'original_prediction': original_prediction,
                'local_prediction': local_prediction,
                'cer': cer,
                'wer': wer,
            }
            (WORK / 'phase1_result.json').write_text(
                json.dumps(result, indent=2) + '\\n', encoding='utf-8'
            )
            print(json.dumps(result, indent=2))
            """),
            md("""
            ## Next Steps

            Na Fazu 2 pređi samo ako je strict load potpun, `assert_close` prolazi i oba
            modela daju isti tekst. Sačuvaj prikaz mouth ROI-a i `phase1_result.json`.
            """),
        ],
    )


def phase2() -> nbf.NotebookNode:
    return notebook(
        "Faza 2 — AI-SPEAK preprocessing po VIPL demo pipeline-u",
        [
            md("""
            ## Goal

            Pretvori `spk*/ser/video_a/*.mp4` u numerisane `128×64` mouth JPEG frejmove
            koristeći brzi BlazeFace detector, isti 68-point face alignment, VIPL afinu
            transformaciju i crop. Ne generiše se manifest, vocab, split ili statičan ROI.
            Neuspesi ostaju u običnom logu.
            """),
            md("""
            ## Setup

            Izaberi T4 GPU. ZIP se jednom kopira sa Drive-a na `/content`; landmark obrada
            sa montiranog Drive-a bi bila znatno sporija. Prilagodi samo `ZIP_ON_DRIVE`.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'face-alignment==1.4.1', 'editdistance>=0.8.1'],
                check=True,
            )
            """),
            code(repo_setup_cell()),
            code("""
            import numpy as np
            import torch
            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            print('GPU:', torch.cuda.get_device_name(0))
            """),
            code(drive_setup_cell()),
            md("## Steps\n\n### 1. Kopiraj i raspakuj lokalni korpus"),
            code("""
            import shutil
            import zipfile

            ZIP_ON_DRIVE = Path('/content/drive/MyDrive/processed.zip')  # promeni po potrebi
            LOCAL_ZIP = Path('/content/processed.zip')
            EXTRACT_ROOT = Path('/content/ai_speak_source')
            OUTPUT_ROOT = Path('/content/ai_speak_lip_blazeface')
            CHECKPOINT_DIR = DRIVE_ROOT / 'phase2_chunks_blazeface'
            RUN_FULL_PREPROCESSING = True

            assert ZIP_ON_DRIVE.exists(), ZIP_ON_DRIVE
            if not LOCAL_ZIP.exists() or not zipfile.is_zipfile(LOCAL_ZIP):
                if LOCAL_ZIP.exists():
                    LOCAL_ZIP.unlink()
                shutil.copy2(ZIP_ON_DRIVE, LOCAL_ZIP)

            EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
            alignment = next(EXTRACT_ROOT.rglob('spk*/alignment/*.align'), None)
            if alignment is None:
                print('AI-SPEAK sadržaj nije pronađen; raspakujem processed.zip...')
                with zipfile.ZipFile(LOCAL_ZIP) as archive:
                    archive.extractall(EXTRACT_ROOT)
                alignment = next(EXTRACT_ROOT.rglob('spk*/alignment/*.align'), None)

            assert alignment is not None, (
                f'Posle raspakivanja nema spk*/alignment/*.align u {EXTRACT_ROOT}. '
                'Proveri strukturu processed.zip arhive.'
            )
            CORPUS_ROOT = alignment.parents[2]
            videos = list(CORPUS_ROOT.glob('spk*/ser/video_a/*.mp4'))
            annotations = list(CORPUS_ROOT.glob('spk*/alignment/*.align'))
            assert videos and len(videos) == len(annotations), (len(videos), len(annotations))
            print('Korpus:', CORPUS_ROOT)
            print('MP4/ALIGN:', len(videos), len(annotations))

            # Restore every completed block after a fresh/disconnected runtime.
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            checkpoint_archives = sorted(CHECKPOINT_DIR.glob('chunk_*.zip'))
            if checkpoint_archives and not next(OUTPUT_ROOT.glob('spk*/video/video_a/*'), None):
                OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
                for index, checkpoint in enumerate(checkpoint_archives, start=1):
                    with zipfile.ZipFile(checkpoint) as archive:
                        archive.extractall(OUTPUT_ROOT)
                    if index % 25 == 0 or index == len(checkpoint_archives):
                        print(f'Vraćeni checkpoint-i: {index}/{len(checkpoint_archives)}')
            restored_samples = len(list(OUTPUT_ROOT.glob('spk*/video/video_a/*')))
            print('Vraćeno/zatečeno gotovih sample foldera:', restored_samples)
            """),
            md("### 2. Prvo uradi jedan GPU smoke primer i upstream `_load_vid` proveru"),
            code("""
            SMOKE_ROOT = Path('/content/ai_speak_lip_blazeface_smoke')
            smoke_command = [
                sys.executable, '-m', 'scripts.prepare_ai_speak',
                '--corpus', str(CORPUS_ROOT), '--output', str(SMOKE_ROOT),
                '--device', 'cuda', '--face-detector', 'blazeface', '--limit', '1',
            ]
            print('Pokrećem:', ' '.join(smoke_command))
            smoke = subprocess.run(smoke_command, text=True, capture_output=True)
            print(smoke.stdout)
            if smoke.returncode:
                print(smoke.stderr)
                raise RuntimeError(
                    f'AI-SPEAK smoke preprocessing nije uspeo (exit={smoke.returncode}). '
                    'Stvarna greška je odštampana neposredno iznad.'
                )

            from lipnet.dataset import MyDataset
            sample_folder = next(SMOKE_ROOT.glob('spk*/video/video_a/*'))
            sample_array = MyDataset._load_vid(sample_folder)
            normalized = sample_array / 255.0
            assert sample_array.shape[1:] == (64, 128, 3)
            assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0
            print('VIPL _load_vid:', sample_array.shape, 'range:', normalized.min(), normalized.max())
            """),
            md("### 3. Obradi ceo korpus na GPU-u i nastavi bez ponavljanja gotovih klipova"),
            code("""
            if RUN_FULL_PREPROCESSING:
                subprocess.run([
                    sys.executable, '-m', 'scripts.prepare_ai_speak',
                    '--corpus', str(CORPUS_ROOT), '--output', str(OUTPUT_ROOT),
                    '--device', 'cuda', '--face-detector', 'blazeface', '--resume',
                    '--report-every', '5',
                    '--checkpoint-dir', str(CHECKPOINT_DIR),
                    '--checkpoint-every', '10',
                ], check=True)
            else:
                print('RUN_FULL_PREPROCESSING=False: ceo GPU posao je namerno preskočen.')
            """),
            md("## Checks\n\n### 4. Vizuelno pregledaj normalne i granične klipove svakog govornika"),
            code("""
            from IPython.display import Image, display

            QA_PATH = OUTPUT_ROOT / 'qa_mouth_crops.jpg'
            FAILURE_LOG = OUTPUT_ROOT / 'failed_clips.log'
            assert QA_PATH.exists(), 'Puna obrada nije napravljena ili nije završena.'
            display(Image(filename=str(QA_PATH)))
            failures = FAILURE_LOG.read_text(encoding='utf-8').strip().splitlines()
            print('Neuspeli klipovi:', len([line for line in failures if line]))
            print('\\n'.join(failures[:20]) if failures else 'Nema neuspelih klipova.')
            print('Ručno potvrdi: usne su u centru, nisu odsečene i crop je stabilan.')
            """),
            md("### 5. Arhiviraj JPEG foldere i logove na Drive"),
            code("""
            if RUN_FULL_PREPROCESSING:
                archive_base = Path('/content/ai_speak_lip')
                archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=OUTPUT_ROOT))
                drive_archive = DRIVE_ROOT / 'ai_speak_lip.zip'
                shutil.copy2(archive, drive_archive)
                print('Sačuvano:', drive_archive, f'{drive_archive.stat().st_size/1024**3:.2f} GiB')
            """),
            md("""
            ## Next Steps

            Ne prelazi na Dataset dok smoke shape/opseg i ručni QA ne prođu. Faza 3 čita samo
            JPEG foldere i originalne `.align` fajlove; `preprocessing.jsonl` i failure log nisu
            ulaz u trening.
            """),
        ],
    )


def phase3() -> nbf.NotebookNode:
    return notebook(
        "Faza 3 — minimalni srpski Dataset adapter",
        [
            md("""
            ## Goal

            Proveri srpski alfabet i tab-separated anotacije, eksplicitni speaker-disjoint
            split, runtime discovery i batch sa dva različito duga klipa. Izlaz ostaje VIPL
            rečnik: `vid`, `txt`, `txt_len`, `vid_len`.
            """),
            md("## Setup\n\nOva faza je lagana i radi na CPU-u. Potrebni su Phase 2 ZIP i originalni `processed.zip`."),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10', 'pytest>=8.4'],
                check=True,
            )
            """),
            code(repo_setup_cell()),
            code(drive_setup_cell()),
            md("### 0. Pokreni lake kompatibilnosne testove (bez model forward-a)"),
            code("""
            subprocess.run(
                [sys.executable, '-m', 'pytest', '-q', 'tests/test_vipl_phases.py'],
                check=True,
            )
            """),
            md("## Steps\n\n### 1. Raspakuj mouth frejmove i samo ALIGN anotacije na lokalni disk"),
            code("""
            import zipfile

            MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            SOURCE_ARCHIVE = Path('/content/drive/MyDrive/processed.zip')  # promeni po potrebi
            MOUTH_ROOT = Path('/content/ai_speak_lip')
            ALIGN_EXTRACT = Path('/content/ai_speak_align')
            assert MOUTH_ARCHIVE.exists() and SOURCE_ARCHIVE.exists()

            if not next(MOUTH_ROOT.glob('spk*/video/video_a/*'), None):
                MOUTH_ROOT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(MOUTH_ARCHIVE) as archive:
                    archive.extractall(MOUTH_ROOT)
            if not next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'), None):
                ALIGN_EXTRACT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
                    members = [
                        member for member in archive.infolist()
                        if '/alignment/' in f'/{member.filename}' and member.filename.endswith('.align')
                    ]
                    archive.extractall(ALIGN_EXTRACT, members=members)
            first_alignment = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'))
            CORPUS_ROOT = first_alignment.parents[2]
            print('Mouth root:', MOUTH_ROOT)
            print('Annotation root:', CORPUS_ROOT)
            """),
            md("### 2. Proveri split, alfabet, parser i CTC round-trip"),
            code("""
            from data.splits import SPLITS, TRAIN_SPEAKERS
            from lipnet.dataset import (
                SERBIAN_LETTERS, SerbianDataset, parse_ai_speak_alignment,
            )

            split_sets = [set(SPLITS[name]) for name in ('train', 'validation', 'test')]
            assert not (split_sets[0] & split_sets[1] | split_sets[0] & split_sets[2] | split_sets[1] & split_sets[2])
            assert 1 + len(SERBIAN_LETTERS) == 29  # blank + 28 vidljivih simbola

            alphabet_text = ''.join(SERBIAN_LETTERS).strip()
            encoded = SerbianDataset.txt2arr(alphabet_text, start=1)
            assert SerbianDataset.arr2txt(encoded, start=1) == alphabet_text

            example_align = next(CORPUS_ROOT.glob(f'{TRAIN_SPEAKERS[0]}/alignment/*.align'))
            example_text = parse_ai_speak_alignment(example_align)
            assert 'sil' not in example_text.split() and 'sp' not in example_text.split()
            print('Klase:', 1 + len(SERBIAN_LETTERS))
            print('Primer:', example_align.name, '->', example_text)
            print('Split:', {name: len(speakers) for name, speakers in SPLITS.items()})
            """),
            md("### 3. Napravi Dataset i batch sa dve različite dužine"),
            code("""
            from torch.utils.data import DataLoader
            from lipnet.dataset import variable_length_collate, validate_ctc_batch

            train_dataset = SerbianDataset(
                video_path=MOUTH_ROOT,
                anno_path=CORPUS_ROOT,
                speakers=TRAIN_SPEAKERS,
                phase='train',
            )
            first = train_dataset[0]
            second = None
            for index in range(1, len(train_dataset)):
                candidate = train_dataset[index]
                if candidate['vid_len'] != first['vid_len']:
                    second = candidate
                    break
            assert second is not None, 'Svi klipovi neočekivano imaju istu dužinu.'
            batch = variable_length_collate([first, second])
            assert set(batch) == {'vid', 'txt', 'txt_len', 'vid_len'}
            assert batch['vid_len'][0] != batch['vid_len'][1]
            assert batch['vid'].shape[2] == int(batch['vid_len'].max())
            validate_ctc_batch(batch, batch['vid_len'])
            print('Broj train primera:', len(train_dataset))
            print({key: tuple(value.shape) for key, value in batch.items()})
            print('vid_len:', batch['vid_len'].tolist(), 'txt_len:', batch['txt_len'].tolist())

            loader = DataLoader(
                train_dataset, batch_size=2, shuffle=False, num_workers=0,
                collate_fn=variable_length_collate,
            )
            assert set(next(iter(loader))) == {'vid', 'txt', 'txt_len', 'vid_len'}
            """),
            md("## Checks\n\n### 4. Proveri normalizaciju, padding i sačuvaj audit"),
            code("""
            import json
            assert 0.0 <= float(batch['vid'].min()) <= float(batch['vid'].max()) <= 1.0
            for index, length in enumerate(batch['vid_len']):
                tail = batch['vid'][index, :, int(length):]
                assert tail.numel() == 0 or float(tail.abs().sum()) == 0.0

            result = {
                'phase': 3,
                'num_classes': 1 + len(SERBIAN_LETTERS),
                'train_samples': len(train_dataset),
                'batch_shape': list(batch['vid'].shape),
                'vid_len': batch['vid_len'].tolist(),
                'txt_len': batch['txt_len'].tolist(),
                'keys': sorted(batch),
            }
            result_path = DRIVE_ROOT / 'phase3_dataset_audit.json'
            result_path.write_text(json.dumps(result, indent=2) + '\\n', encoding='utf-8')
            print(json.dumps(result, indent=2))
            """),
            md("""
            ## Next Steps

            Faza 3 prolazi kada su govornici disjunktni, round-trip čuva sva slova, dva
            različita klipa dele padded batch i CTC feasibility provera ne prijavi grešku.
            Faza 4 koristi isti batch ugovor na GPU-u.
            """),
        ],
    )


def phase4() -> nbf.NotebookNode:
    return notebook(
        "Faza 4 — srpski LipNet, transfer checkpoint-a i CTC backward",
        [
            md("""
            ## Goal

            Instanciraj 29-klasni srpski LipNet, prenesi svaki kompatibilan VIPL backbone/BiGRU
            parametar, dokaži da su preskočeni samo `FC.weight` i `FC.bias`, pa uradi jedan
            forward + CTC loss + backward na malom realnom batch-u. Nema trening epohe.
            """),
            md("""
            ## Setup

            Izaberi T4 GPU. Ovaj notebook namerno ne pokreće optimizer niti fine-tuning;
            to počinje tek u Fazi 5.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10'],
                check=True,
            )
            """),
            code(repo_setup_cell()),
            code("""
            import numpy as np
            import torch
            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            DEVICE = torch.device('cuda')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            print('GPU:', torch.cuda.get_device_name(0))
            """),
            code(drive_setup_cell()),
            md("## Steps\n\n### 1. Učitaj Phase 2 frejmove i anotacije"),
            code("""
            import zipfile

            MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            SOURCE_ARCHIVE = Path('/content/drive/MyDrive/processed.zip')  # promeni po potrebi
            MOUTH_ROOT = Path('/content/ai_speak_lip')
            ALIGN_EXTRACT = Path('/content/ai_speak_align')
            assert MOUTH_ARCHIVE.exists() and SOURCE_ARCHIVE.exists()
            if not next(MOUTH_ROOT.glob('spk*/video/video_a/*'), None):
                MOUTH_ROOT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(MOUTH_ARCHIVE) as archive:
                    archive.extractall(MOUTH_ROOT)
            if not next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'), None):
                ALIGN_EXTRACT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
                    members = [member for member in archive.infolist()
                               if '/alignment/' in f'/{member.filename}' and member.filename.endswith('.align')]
                    archive.extractall(ALIGN_EXTRACT, members=members)
            CORPUS_ROOT = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align')).parents[2]
            """),
            md("### 2. Preuzmi pinovani engleski checkpoint i audituj transfer"),
            code(f"""
            from urllib.request import urlretrieve

            from lipnet.dataset import SERBIAN_LETTERS
            from lipnet.model import LipNet
            from lipnet.train import load_vipl_transfer

            UPSTREAM_SHA = {UPSTREAM_SHA!r}
            CHECKPOINT = Path('/content') / {CHECKPOINT_NAME!r}
            CHECKPOINT_URL = (
                'https://raw.githubusercontent.com/VIPL-Audio-Visual-Speech-Understanding/'
                f'LipNet-PyTorch/{{UPSTREAM_SHA}}/pretrain/{CHECKPOINT_NAME}'
            )
            if not CHECKPOINT.exists():
                urlretrieve(CHECKPOINT_URL, CHECKPOINT)

            NUM_CLASSES = 1 + len(SERBIAN_LETTERS)
            assert NUM_CLASSES == 29
            model = LipNet(num_classes=NUM_CLASSES).to(DEVICE)
            audit = load_vipl_transfer(model, CHECKPOINT)
            assert set(audit.skipped_shape) == {'FC.weight', 'FC.bias'}
            assert not audit.missing_in_checkpoint and not audit.unexpected_in_checkpoint
            assert all(name in audit.loaded for name in ('conv1.weight', 'conv2.weight', 'conv3.weight'))
            assert any(name.startswith('gru1.') for name in audit.loaded)
            assert any(name.startswith('gru2.') for name in audit.loaded)
            print(audit.summary())
            """),
            md("### 3. Izaberi dva kratka realna klipa koja zadovoljavaju CTC i napravi batch"),
            code("""
            from data.splits import TRAIN_SPEAKERS
            from lipnet.dataset import SerbianDataset, minimum_ctc_steps, variable_length_collate

            dataset = SerbianDataset(MOUTH_ROOT, CORPUS_ROOT, TRAIN_SPEAKERS, phase='train')
            ordered_indices = sorted(
                range(len(dataset)),
                key=lambda index: len(list(dataset.data[index][0].glob('*.jpg'))),
            )
            samples = []
            for index in ordered_indices:
                sample = dataset[index]
                target = sample['txt'][:sample['txt_len']]
                if minimum_ctc_steps(target) <= sample['vid_len']:
                    samples.append(sample)
                if len(samples) == 2:
                    break
            assert len(samples) == 2, 'Nisu pronađena dva CTC-validna klipa.'
            batch = variable_length_collate(samples)
            print('Batch:', tuple(batch['vid'].shape))
            print('vid_len:', batch['vid_len'].tolist(), 'txt_len:', batch['txt_len'].tolist())
            """),
            md("## Checks\n\n### 4. Jedan GPU forward, konačan CTC loss i backward audit"),
            code("""
            from lipnet.train import backward_smoke_step

            loss, logits_shape = backward_smoke_step(model, batch, DEVICE)
            assert logits_shape[0] == 2 and logits_shape[-1] == NUM_CLASSES
            assert np.isfinite(loss)
            print('Logits (B,T,C):', logits_shape)
            print('CTC loss:', loss)
            print('PASS: svi trainable parametri imaju konačne gradijente.')
            """),
            code("""
            import json

            result = {
                'phase': 4,
                'upstream_commit': UPSTREAM_SHA,
                'num_classes': NUM_CLASSES,
                'loaded_tensors': len(audit.loaded),
                'skipped_shape': list(audit.skipped_shape),
                'missing_in_checkpoint': list(audit.missing_in_checkpoint),
                'unexpected_in_checkpoint': list(audit.unexpected_in_checkpoint),
                'batch_shape': list(batch['vid'].shape),
                'logits_shape': list(logits_shape),
                'ctc_loss': loss,
                'backward': 'passed',
            }
            result_path = DRIVE_ROOT / 'phase4_transfer_ctc_audit.json'
            result_path.write_text(json.dumps(result, indent=2) + '\\n', encoding='utf-8')
            print(json.dumps(result, indent=2))
            """),
            md("""
            ## Next Steps

            Faza 4 je završena samo ako audit preskače tačno dva FC tenzora, CTC loss je
            konačan i svi gradijenti su konačni. Tek tada Faza 5 sme da uvede optimizer,
            checkpoint-e i baseline fine-tuning.
            """),
        ],
    )


def phase5() -> nbf.NotebookNode:
    return notebook(
        "Faza 5 — baseline fine-tuning srpskog LipNet-a",
        [
            md("""
            ## Goal

            Fine-tune-uj preneti VIPL model na speaker-disjoint AI-SPEAK splitu. Prve tri
            epohe uče samo novi srpski head, zatim se odmrzava ceo model. Najbolji checkpoint
            bira se isključivo strogo manjim validation WER-om; test govornici se evaluiraju
            tek nakon izbora modela.
            """),
            md("""
            ## Setup

            Izaberi **Runtime → Change runtime type → T4 GPU**. Notebook čuva `latest.pt`,
            `best.pt`, istoriju i rezultate u `MyDrive/LipNet/phase5`, pa bezbedno nastavlja
            rad posle prekida Colab sesije.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10',
                 'nbformat>=5.10.4', 'pytest>=8.4'],
                check=True,
            )
            """),
            code(repo_setup_cell()),
            code("""
            import json
            import random
            from dataclasses import asdict

            import matplotlib.pyplot as plt
            import numpy as np
            import torch

            from lipnet.train import FineTuneConfig

            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            DEVICE = torch.device('cuda')
            CONFIG = FineTuneConfig(
                max_epochs=30,
                warmup_epochs=3,
                early_stopping_patience=5,
                batch_size=2,
                backbone_lr=2e-5,
                head_lr=1e-4,
                num_workers=2,
                random_seed=0,
            )
            random.seed(CONFIG.random_seed)
            np.random.seed(CONFIG.random_seed)
            torch.manual_seed(CONFIG.random_seed)
            torch.cuda.manual_seed_all(CONFIG.random_seed)
            print('GPU:', torch.cuda.get_device_name(0))
            print('Konfiguracija:', json.dumps(asdict(CONFIG), indent=2))
            """),
            code(drive_setup_cell()),
            code("""
            PHASE5_DIR = DRIVE_ROOT / 'phase5'
            PHASE5_DIR.mkdir(parents=True, exist_ok=True)
            LATEST_CHECKPOINT = PHASE5_DIR / 'latest.pt'
            BEST_CHECKPOINT = PHASE5_DIR / 'best.pt'
            HISTORY_PATH = PHASE5_DIR / 'history.json'
            RESULTS_PATH = PHASE5_DIR / 'results.json'
            CURVES_PATH = PHASE5_DIR / 'training_curves.png'
            print('Phase 5 izlaz:', PHASE5_DIR)
            """),
            md("### 0. Pokreni CPU kompatibilnosne testove"),
            code("""
            subprocess.run(
                [sys.executable, '-m', 'pytest', '-q',
                 'tests/test_vipl_phases.py', 'tests/test_phase5.py'],
                check=True,
            )
            """),
            md("## Steps\n\n### 1. Raspakuj mouth frejmove i ALIGN anotacije na lokalni disk"),
            code("""
            import zipfile

            MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            SOURCE_ARCHIVE = Path('/content/drive/MyDrive/processed.zip')  # promeni po potrebi
            MOUTH_ROOT = Path('/content/ai_speak_lip')
            ALIGN_EXTRACT = Path('/content/ai_speak_align')
            assert MOUTH_ARCHIVE.exists(), MOUTH_ARCHIVE
            assert SOURCE_ARCHIVE.exists(), SOURCE_ARCHIVE

            if not next(MOUTH_ROOT.glob('spk*/video/video_a/*'), None):
                MOUTH_ROOT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(MOUTH_ARCHIVE) as archive:
                    archive.extractall(MOUTH_ROOT)
            if not next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'), None):
                ALIGN_EXTRACT.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
                    members = [
                        member for member in archive.infolist()
                        if '/alignment/' in f'/{member.filename}'
                        and member.filename.endswith('.align')
                    ]
                    archive.extractall(ALIGN_EXTRACT, members=members)
            first_alignment = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'))
            CORPUS_ROOT = first_alignment.parents[2]
            print('Mouth root:', MOUTH_ROOT)
            print('Annotation root:', CORPUS_ROOT)
            """),
            md("### 2. Formiraj splitove i runtime CTC filter bez manifesta"),
            code("""
            from torch.utils.data import DataLoader, Subset

            from data.splits import (
                SPLITS, TEST_SPEAKERS, TRAIN_SPEAKERS, VALIDATION_SPEAKERS,
            )
            from lipnet.dataset import SerbianDataset, variable_length_collate
            from lipnet.train import scan_ctc_compatibility

            raw_datasets = {
                'train': SerbianDataset(MOUTH_ROOT, CORPUS_ROOT, TRAIN_SPEAKERS, phase='train'),
                'validation': SerbianDataset(
                    MOUTH_ROOT, CORPUS_ROOT, VALIDATION_SPEAKERS, phase='validation'
                ),
                'test': SerbianDataset(MOUTH_ROOT, CORPUS_ROOT, TEST_SPEAKERS, phase='test'),
            }
            ctc_reports = {
                name: scan_ctc_compatibility(dataset)
                for name, dataset in raw_datasets.items()
            }
            datasets = {
                name: Subset(raw_datasets[name], report.valid_indices)
                for name, report in ctc_reports.items()
            }
            for name, report in ctc_reports.items():
                print(
                    f'{name:10s}: valid={report.valid_count}, '
                    f'CTC-odbačeno={report.invalid_count}'
                )

            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % (2**32)
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            def make_loader(name, *, epoch=0, shuffle=False):
                generator = torch.Generator()
                generator.manual_seed(CONFIG.random_seed + epoch)
                return DataLoader(
                    datasets[name],
                    batch_size=CONFIG.batch_size,
                    shuffle=shuffle,
                    num_workers=CONFIG.num_workers,
                    collate_fn=variable_length_collate,
                    pin_memory=True,
                    worker_init_fn=seed_worker,
                    generator=generator,
                    persistent_workers=False,
                )
            """),
            md("### 3. Učitaj transfer ili automatski nastavi `latest.pt`"),
            code(f"""
            from urllib.request import urlretrieve

            from lipnet.dataset import SERBIAN_LETTERS
            from lipnet.model import LipNet
            from lipnet.train import (
                build_finetune_optimizer, greedy_decode, load_training_checkpoint,
                load_vipl_transfer, set_backbone_trainable,
            )

            UPSTREAM_SHA = {UPSTREAM_SHA!r}
            VIPL_CHECKPOINT = Path('/content') / {CHECKPOINT_NAME!r}
            VIPL_CHECKPOINT_URL = (
                'https://raw.githubusercontent.com/VIPL-Audio-Visual-Speech-Understanding/'
                f'LipNet-PyTorch/{{UPSTREAM_SHA}}/pretrain/{CHECKPOINT_NAME}'
            )
            if not VIPL_CHECKPOINT.exists():
                urlretrieve(VIPL_CHECKPOINT_URL, VIPL_CHECKPOINT)

            NUM_CLASSES = 1 + len(SERBIAN_LETTERS)
            assert NUM_CLASSES == 29
            model = LipNet(num_classes=NUM_CLASSES).to(DEVICE)
            optimizer = build_finetune_optimizer(
                model, backbone_lr=CONFIG.backbone_lr, head_lr=CONFIG.head_lr
            )

            if LATEST_CHECKPOINT.exists():
                state = load_training_checkpoint(LATEST_CHECKPOINT, model, optimizer)
                assert state.config == asdict(CONFIG), (
                    'Resume konfiguracija se razlikuje od CONFIG ćelije: '
                    f'checkpoint={{state.config}} current={{asdict(CONFIG)}}'
                )
                start_epoch = state.next_epoch
                best_val_wer = state.best_val_wer
                best_epoch = state.best_epoch
                epochs_without_improvement = state.epochs_without_improvement
                history = list(state.history)
                audit = None
                print('Nastavljam od epohe', start_epoch + 1)
            else:
                audit = load_vipl_transfer(model, VIPL_CHECKPOINT)
                assert set(audit.skipped_shape) == {{'FC.weight', 'FC.bias'}}
                assert not audit.missing_in_checkpoint and not audit.unexpected_in_checkpoint
                start_epoch = 0
                best_val_wer = float('inf')
                best_epoch = -1
                epochs_without_improvement = 0
                history = []
                print('Novi trening:', audit.summary())

            set_backbone_trainable(model, start_epoch >= CONFIG.warmup_epochs)
            """),
            md("### 4. Treniraj, validiraj i čuvaj latest/best checkpoint"),
            code("""
            from lipnet.train import (
                run_epoch, save_training_checkpoint, validation_wer_improved,
            )

            checkpoint_metadata = {
                'phase': 5,
                'upstream_commit': UPSTREAM_SHA,
                'num_classes': NUM_CLASSES,
                'speakers': {name: list(value) for name, value in SPLITS.items()},
                'ctc_filter': {
                    name: {'valid': report.valid_count, 'invalid': report.invalid_count}
                    for name, report in ctc_reports.items()
                },
            }

            for epoch in range(start_epoch, CONFIG.max_epochs):
                backbone_trainable = epoch >= CONFIG.warmup_epochs
                set_backbone_trainable(model, backbone_trainable)
                train_result = run_epoch(
                    model,
                    make_loader('train', epoch=epoch, shuffle=True),
                    DEVICE,
                    optimizer,
                )
                validation_result = run_epoch(
                    model, make_loader('validation'), DEVICE
                )
                improved = validation_wer_improved(validation_result.wer, best_val_wer)
                if improved:
                    best_val_wer = validation_result.wer
                    best_epoch = epoch

                if epoch < CONFIG.warmup_epochs:
                    epochs_without_improvement = 0
                elif improved:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                record = {
                    'epoch': epoch,
                    'stage': 'full' if backbone_trainable else 'head_only',
                    'train': train_result.metrics(),
                    'validation': validation_result.metrics(),
                }
                history.append(record)
                checkpoint_arguments = dict(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_wer=best_val_wer,
                    best_epoch=best_epoch,
                    epochs_without_improvement=epochs_without_improvement,
                    history=history,
                    config=CONFIG,
                    metadata=checkpoint_metadata,
                )
                if improved:
                    save_training_checkpoint(BEST_CHECKPOINT, **checkpoint_arguments)
                save_training_checkpoint(LATEST_CHECKPOINT, **checkpoint_arguments)
                HISTORY_PATH.write_text(
                    json.dumps(history, indent=2, ensure_ascii=False) + '\\n',
                    encoding='utf-8',
                )

                print(
                    f'Epoha {{epoch + 1:02d}}/{{CONFIG.max_epochs}} '
                    f'[{{record["stage"]}}] '
                    f'train loss={{train_result.loss:.4f}} WER={{train_result.wer:.4f}} | '
                    f'val loss={{validation_result.loss:.4f}} '
                    f'WER={{validation_result.wer:.4f}} CER={{validation_result.cer:.4f}} '
                    f'exact={{validation_result.sentence_exact_match:.4f}}'
                    + ('  *BEST*' if improved else '')
                )
                if (
                    epoch >= CONFIG.warmup_epochs
                    and epochs_without_improvement >= CONFIG.early_stopping_patience
                ):
                    print('Early stopping: validation WER se nije poboljšao.')
                    break

            assert BEST_CHECKPOINT.exists(), 'Nije sačuvan najbolji checkpoint.'
            print('Najbolja epoha:', best_epoch + 1, 'validation WER:', best_val_wer)
            """),
            md("### 5. Prikaži i sačuvaj krive treniranja"),
            code("""
            epochs = [item['epoch'] + 1 for item in history]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for axis, metric, title in zip(
                axes, ('loss', 'wer', 'cer'), ('CTC loss', 'WER', 'CER')
            ):
                axis.plot(epochs, [item['train'][metric] for item in history], label='train')
                axis.plot(
                    epochs,
                    [item['validation'][metric] for item in history],
                    label='validation',
                )
                axis.axvline(CONFIG.warmup_epochs, color='gray', linestyle='--', alpha=0.7)
                axis.set(title=title, xlabel='Epoha', ylabel=metric.upper())
                axis.grid(alpha=0.25)
                axis.legend()
            plt.tight_layout()
            fig.savefig(CURVES_PATH, dpi=160, bbox_inches='tight')
            plt.show()
            """),
            md("## Checks\n\n### 6. Učitaj `best.pt` i evaluiraj neviđene test govornike jednom"),
            code("""
            import hashlib

            from lipnet.train import load_training_checkpoint

            best_state = load_training_checkpoint(
                BEST_CHECKPOINT, model, optimizer, restore_rng=False
            )
            assert best_state.best_epoch == best_state.next_epoch - 1
            best_validation = next(
                item['validation']
                for item in best_state.history
                if item['epoch'] == best_state.best_epoch
            )
            checkpoint_hasher = hashlib.sha256()
            with BEST_CHECKPOINT.open('rb') as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b''):
                    checkpoint_hasher.update(chunk)
            best_checkpoint_sha256 = checkpoint_hasher.hexdigest()
            FORCE_TEST_REEVALUATION = False
            previous_results = None
            if RESULTS_PATH.exists():
                previous_results = json.loads(RESULTS_PATH.read_text(encoding='utf-8'))

            if (
                previous_results is not None
                and previous_results.get('best_epoch') == best_state.best_epoch
                and previous_results.get('best_checkpoint_sha256') == best_checkpoint_sha256
                and not FORCE_TEST_REEVALUATION
            ):
                results = previous_results
                print('Koristim već sačuvanu test evaluaciju za isti best checkpoint.')
            else:
                test_result = run_epoch(model, make_loader('test'), DEVICE)
                qualitative = [
                    {'reference': reference, 'prediction': prediction}
                    for reference, prediction in list(
                        zip(test_result.references, test_result.predictions)
                    )[:10]
                ]
                results = {
                    'phase': 5,
                    'upstream_commit': UPSTREAM_SHA,
                    'config': asdict(CONFIG),
                    'best_epoch': best_state.best_epoch,
                    'best_validation_wer': best_state.best_val_wer,
                    'best_checkpoint_sha256': best_checkpoint_sha256,
                    'validation': best_validation,
                    'test': test_result.metrics(),
                    'test_samples': test_result.samples,
                    'qualitative': qualitative,
                    'ctc_filter': checkpoint_metadata['ctc_filter'],
                }
                RESULTS_PATH.write_text(
                    json.dumps(results, indent=2, ensure_ascii=False) + '\\n',
                    encoding='utf-8',
                )

            print(json.dumps({
                'best_epoch': results['best_epoch'] + 1,
                'validation': results['validation'],
                'test': results['test'],
                'test_samples': results['test_samples'],
            }, indent=2, ensure_ascii=False))
            for item in results['qualitative']:
                print('REF :', item['reference'])
                print('PRED:', item['prediction'])
                print()
            """),
            md("### 7. Potvrdi checkpoint round-trip i inference ugovor"),
            code("""
            reloaded_model = LipNet(num_classes=NUM_CLASSES).to(DEVICE)
            reloaded_optimizer = build_finetune_optimizer(
                reloaded_model,
                backbone_lr=CONFIG.backbone_lr,
                head_lr=CONFIG.head_lr,
            )
            reloaded_state = load_training_checkpoint(
                BEST_CHECKPOINT, reloaded_model, reloaded_optimizer, restore_rng=False
            )
            inference_batch = next(iter(make_loader('test')))
            reloaded_model.eval()
            with torch.no_grad():
                inference_logits = reloaded_model(inference_batch['vid'].to(DEVICE))
            inference_text = greedy_decode(
                inference_logits, output_lengths=inference_batch['vid_len']
            )
            assert len(inference_text) == inference_batch['vid'].shape[0]
            assert reloaded_state.best_epoch == results['best_epoch']
            print('PASS: best checkpoint se ponovo učitava i daje inference:', inference_text)
            """),
            md("""
            ## Next Steps

            Faza 5 je završena kada postoje `best.pt`, `latest.pt`, istorija, krive i test
            rezultat za neviđene govornike. Faza 6 mora koristiti isti `best.pt`, isti test
            split i isti greedy dekoder; menja se samo ulazna rezolucija, blur ili crop jitter.
            """),
        ],
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "00_faza_0_upstream_restart.ipynb": phase0(),
        "01_faza_1_grid_parity.ipynb": phase1(),
        "02_faza_2_ai_speak_preprocessing.ipynb": phase2(),
        "03_faza_3_serbian_dataset.ipynb": phase3(),
        "04_faza_4_transfer_ctc_smoke.ipynb": phase4(),
        "05_faza_5_baseline_finetuning.ipynb": phase5(),
    }
    for name, value in notebooks.items():
        nbf.write(value, OUTPUT / name)
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
