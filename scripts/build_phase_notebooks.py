#!/usr/bin/env python3
"""Generate the reader-facing Colab notebooks for phases 0 through 8."""

from __future__ import annotations

import argparse
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


def notebook(title: str, cells: list, *, gpu: bool = False) -> nbf.NotebookNode:
    metadata = {
        "colab": {"name": title, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    }
    if gpu:
        metadata["accelerator"] = "GPU"
    value = nbf.v4.new_notebook(
        cells=[md(f"# {title}")] + cells,
        metadata=metadata,
    )
    for index, cell in enumerate(value.cells):
        cell.id = f"cell-{index:02d}"
    return value


def repo_setup_cell(*, refresh_imports: bool = False) -> str:
    if not refresh_imports:
        return f"""
        import os
        import subprocess
        from pathlib import Path

        REPO_URL = {REPO_URL!r}
        REPO = Path('/content/lipnet-serbian')
        if not (REPO / 'lipnet').exists():
            subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
        else:
            subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
        os.chdir(REPO)
        print('Repo:', REPO)
        print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        """

    return f"""
    import importlib
    import os
    import subprocess
    import sys
    from pathlib import Path

    REPO_URL = {REPO_URL!r}
    REPO = Path('/content/lipnet-serbian')
    if not (REPO / 'lipnet').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(REPO)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
    os.chdir(REPO)

    # A git pull updates source files, but an active Colab kernel can still hold
    # old lipnet modules in sys.modules.  Clear only this project's modules so
    # all later imports use one coherent checkout.
    stale_modules = [
        name for name in sys.modules
        if name == 'lipnet' or name.startswith('lipnet.')
    ]
    for name in stale_modules:
        del sys.modules[name]
    repo_path = str(REPO)
    sys.path[:] = [entry for entry in sys.path if entry != repo_path]
    sys.path.insert(0, repo_path)
    importlib.invalidate_caches()

    print('Repo:', REPO)
    print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    if stale_modules:
        print('Osveženi Python moduli:', ', '.join(sorted(stale_modules)))
    """


def drive_setup_cell(*, force_remount: bool = False) -> str:
    force_argument = ", force_remount=True" if force_remount else ""
    return f"""
    from google.colab import drive

    drive.mount('/content/drive'{force_argument})
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
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
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
        gpu=True,
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

            Završeni `MyDrive/LipNet/ai_speak_lip.zip` već postoji. Smoke i puna BlazeFace
            obrada su zato podrazumevano isključeni; notebook ostaje kao evidencija toka.
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
            print(
                'GPU:', torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else 'nije potreban za reuse postojećeg ZIP-a'
            )
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
            RUN_SMOKE_PREPROCESSING = False
            RUN_FULL_PREPROCESSING = False
            EXISTING_MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            assert EXISTING_MOUTH_ARCHIVE.exists(), EXISTING_MOUTH_ARCHIVE
            print('Zamrznuti mouth artefakt:', EXISTING_MOUTH_ARCHIVE)

            CORPUS_ROOT = None
            videos = []
            annotations = []
            if RUN_SMOKE_PREPROCESSING or RUN_FULL_PREPROCESSING:
                assert ZIP_ON_DRIVE.exists(), ZIP_ON_DRIVE
                if not LOCAL_ZIP.exists() or not zipfile.is_zipfile(LOCAL_ZIP):
                    if LOCAL_ZIP.exists():
                        LOCAL_ZIP.unlink()
                    shutil.copy2(ZIP_ON_DRIVE, LOCAL_ZIP)
                EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
                alignment = next(EXTRACT_ROOT.rglob('spk*/alignment/*.align'), None)
                if alignment is None:
                    with zipfile.ZipFile(LOCAL_ZIP) as archive:
                        archive.extractall(EXTRACT_ROOT)
                    alignment = next(EXTRACT_ROOT.rglob('spk*/alignment/*.align'), None)
                assert alignment is not None
                CORPUS_ROOT = alignment.parents[2]
                videos = list(CORPUS_ROOT.glob('spk*/ser/video_a/*.mp4'))
                annotations = list(CORPUS_ROOT.glob('spk*/alignment/*.align'))
                assert videos and len(videos) == len(annotations)
                CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                print('Korpus MP4/ALIGN:', len(videos), len(annotations))
            else:
                print('processed.zip se ne kopira niti raspakuje.')
            """),
            md("### 2. Prvo uradi jedan GPU smoke primer i upstream `_load_vid` proveru"),
            code("""
            SMOKE_ROOT = Path('/content/ai_speak_lip_blazeface_smoke')
            if RUN_SMOKE_PREPROCESSING:
                smoke_command = [
                    sys.executable, '-m', 'scripts.prepare_ai_speak',
                    '--corpus', str(CORPUS_ROOT), '--output', str(SMOKE_ROOT),
                    '--device', 'cuda', '--face-detector', 'blazeface', '--limit', '1',
                ]
                smoke = subprocess.run(smoke_command, text=True, capture_output=True)
                print(smoke.stdout)
                if smoke.returncode:
                    print(smoke.stderr)
                    raise RuntimeError(f'Smoke preprocessing nije uspeo: {smoke.returncode}')
                from lipnet.dataset import MyDataset
                sample_folder = next(SMOKE_ROOT.glob('spk*/video/video_a/*'))
                sample_array = MyDataset._load_vid(sample_folder)
                normalized = sample_array / 255.0
                assert sample_array.shape[1:] == (64, 128, 3)
                assert 0.0 <= float(normalized.min()) <= float(normalized.max()) <= 1.0
                print('VIPL _load_vid:', sample_array.shape, 'range:', normalized.min(), normalized.max())
            else:
                print('RUN_SMOKE_PREPROCESSING=False: BlazeFace smoke je preskočen.')
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
            md("## Checks\n\n### 4. Proveri potpunost i vizuelno pregledaj granične klipove"),
            code("""
            assert EXISTING_MOUTH_ARCHIVE.stat().st_size > 0
            if RUN_FULL_PREPROCESSING:
                from IPython.display import Image, display
                QA_PATH = OUTPUT_ROOT / 'qa_mouth_crops.jpg'
                FAILURE_LOG = OUTPUT_ROOT / 'failed_clips.log'
                PREPROCESSING_LOG = OUTPUT_ROOT / 'preprocessing.jsonl'
                assert QA_PATH.exists() and FAILURE_LOG.exists() and PREPROCESSING_LOG.exists()
                display(Image(filename=str(QA_PATH)))
                print('Nova obrada zahteva ručnu QA potvrdu pre arhiviranja.')
            else:
                print('Postojeći ai_speak_lip.zip je potvrđen; QA i BlazeFace se ne ponavljaju.')
            """),
            md("### 5. Arhiviraj JPEG foldere i logove na Drive"),
            code("""
            if RUN_FULL_PREPROCESSING:
                MANUAL_QA_PASSED = False  # samo za namerno novu punu obradu
                assert MANUAL_QA_PASSED, 'Ručno pregledaj QA sliku pre nove arhive.'
                archive_base = Path('/content/ai_speak_lip')
                archive = Path(shutil.make_archive(str(archive_base), 'zip', root_dir=OUTPUT_ROOT))
                drive_archive = DRIVE_ROOT / 'ai_speak_lip.zip'
                shutil.copy2(archive, drive_archive)
                print('Sačuvano:', drive_archive, f'{drive_archive.stat().st_size/1024**3:.2f} GiB')
            else:
                print('Postojeći ai_speak_lip.zip ostaje neizmenjen.')
            """),
            md("""
            ## Next Steps

            U nastavku projekta ne pokretati preprocessing ponovo. Faze 3–7 samo raspakuju
            postojeći `ai_speak_lip.zip`; logovi nisu ulaz u trening.
            """),
        ],
        gpu=True,
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

            alphabet_text = ' '.join(('a', ''.join(SERBIAN_LETTERS[1:])))
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

            datasets_by_split = {
                name: SerbianDataset(
                    video_path=MOUTH_ROOT,
                    anno_path=CORPUS_ROOT,
                    speakers=speakers,
                    phase=name,
                )
                for name, speakers in SPLITS.items()
            }
            expected_total = 0
            for name, dataset in datasets_by_split.items():
                discovered = {(speaker, sample_id) for _, speaker, sample_id in dataset.data}
                expected = {
                    (speaker, path.stem)
                    for speaker in SPLITS[name]
                    for path in (CORPUS_ROOT / speaker / 'alignment').glob('*.align')
                }
                expected_total += len(expected)
                assert discovered == expected, (
                    f'{name}: očekivano={len(expected)} pronađeno={len(discovered)}'
                )
            assert sum(map(len, datasets_by_split.values())) == expected_total
            train_dataset = datasets_by_split['train']
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
                'samples_by_split': {
                    name: len(dataset) for name, dataset in datasets_by_split.items()
                },
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
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
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
            assert set(audit.skipped_shape) == {{'FC.weight', 'FC.bias'}}
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
                'torch_version': torch.__version__,
                'gpu': torch.cuda.get_device_name(0),
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
        gpu=True,
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
            `best.pt`, istoriju i rezultate u `MyDrive/LipNet/phase5_length_aware_v2`, pa bezbedno nastavlja
            rad posle prekida Colab sesije.

            Folder ima novu oznaku zato što stari checkpoint-i nisu length-aware i ne smeju
            se nastaviti niti porediti kao da koriste isti trening protokol.

            Početni limit od 30 epoha produžen je na 45 jer je 30. epoha dala novi najbolji
            validation WER. Pri nastavku je dozvoljeno samo povećanje `max_epochs`; svi ostali
            hiperparametri moraju ostati identični checkpoint-u.
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
            code(repo_setup_cell(refresh_imports=True)),
            code("""
            import inspect
            import json
            import random
            from dataclasses import asdict

            import matplotlib.pyplot as plt
            import numpy as np
            import torch

            from lipnet.model import LipNet
            from lipnet.train import FineTuneConfig

            forward_parameters = inspect.signature(LipNet.forward).parameters
            assert 'lengths' in forward_parameters, (
                'Učitana LipNet klasa nije length-aware. Ponovo pokreni Setup ćelije; '
                f'model={inspect.getfile(LipNet)}, signature={inspect.signature(LipNet.forward)}'
            )
            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            DEVICE = torch.device('cuda')
            CONFIG = FineTuneConfig(
                max_epochs=45,
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
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print('LipNet:', inspect.getfile(LipNet), inspect.signature(LipNet.forward))
            print('GPU:', torch.cuda.get_device_name(0))
            print('Konfiguracija:', json.dumps(asdict(CONFIG), indent=2))
            """),
            code(drive_setup_cell()),
            code("""
            TRAINING_PROTOCOL = 'length-aware-bigru-corpus-metrics-v2'
            PHASE5_DIR = DRIVE_ROOT / 'phase5_length_aware_v2'
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
            for name, dataset in raw_datasets.items():
                expected = {
                    (speaker, path.stem)
                    for speaker in SPLITS[name]
                    for path in (CORPUS_ROOT / speaker / 'alignment').glob('*.align')
                }
                discovered = {(speaker, sample_id) for _, speaker, sample_id in dataset.data}
                assert discovered == expected, (
                    f'{name}: očekivano={len(expected)} pronađeno={len(discovered)}'
                )
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
                load_vipl_transfer, set_backbone_trainable, validate_resume_config,
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
                validate_resume_config(state.config, CONFIG)
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
                'training_protocol': TRAINING_PROTOCOL,
                'upstream_commit': UPSTREAM_SHA,
                'num_classes': NUM_CLASSES,
                'speakers': {name: list(value) for name, value in SPLITS.items()},
                'ctc_filter': {
                    name: {'valid': report.valid_count, 'invalid': report.invalid_count}
                    for name, report in ctc_reports.items()
                },
                'environment': {
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0),
                },
            }

            assert 'lengths' in inspect.signature(model.forward).parameters, (
                'Model u memoriji nije length-aware; pokreni notebook ponovo od Setup sekcije.'
            )
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
                    f'Epoha {epoch + 1:02d}/{CONFIG.max_epochs} '
                    f'[{record["stage"]}] '
                    f'train loss={train_result.loss:.4f} WER={train_result.wer:.4f} | '
                    f'val loss={validation_result.loss:.4f} '
                    f'WER={validation_result.wer:.4f} CER={validation_result.cer:.4f} '
                    f'exact={validation_result.sentence_exact_match:.4f}'
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
                and previous_results.get('metrics_definition') == 'corpus-edit-distance-v1'
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
                    'training_protocol': TRAINING_PROTOCOL,
                    'metrics_definition': 'corpus-edit-distance-v1',
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
                    'environment': checkpoint_metadata['environment'],
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
                inference_logits = reloaded_model(
                    inference_batch['vid'].to(DEVICE),
                    lengths=inference_batch['vid_len'],
                )
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
        gpu=True,
    )


def phase6() -> nbf.NotebookNode:
    return notebook(
        "Faza 6 — provera rezultata i robustnost ulaza",
        [
            md("""
            ## Goal

            Ponovo izračunaj baseline test metrike iz `best.pt`, potvrdi ih prema Phase 5
            rezultatu i zatim, bez dodatnog treninga, izmeri uticaj niže rezolucije, blur-a
            i kontrolisanog crop pomeranja. Sve varijante koriste isti test split, isti
            checkpoint, isti greedy CTC dekoder i corpus-level WER/CER.
            """),
            md("""
            ## Setup

            Izaberi T4 GPU. Pre ovog notebooka mora biti završen novi length-aware notebook
            Faze 5 i na Drive-u moraju postojati `phase5_length_aware_v2/best.pt` i
            `phase5_length_aware_v2/results.json`.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10'],
                check=True,
            )
            """),
            code(repo_setup_cell(refresh_imports=True)),
            code("""
            import hashlib
            import inspect
            import json
            import math

            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            import torch.nn.functional as F

            from lipnet.model import LipNet

            forward_parameters = inspect.signature(LipNet.forward).parameters
            assert 'lengths' in forward_parameters, (
                'Učitana LipNet klasa nije length-aware. Ponovo pokreni Setup ćelije; '
                f'model={inspect.getfile(LipNet)}, signature={inspect.signature(LipNet.forward)}'
            )
            assert torch.cuda.is_available(), 'Uključi T4 GPU u Colab Runtime postavkama.'
            DEVICE = torch.device('cuda')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print('LipNet:', inspect.getfile(LipNet), inspect.signature(LipNet.forward))
            print('GPU:', torch.cuda.get_device_name(0))
            """),
            code(drive_setup_cell()),
            code("""
            # Faza 7 moze da se pokrene pre nego sto su novi decoder moduli objavljeni
            # na GitHub-u. Kopije na Drive-u su mali izvorni fajlovi; mouth frejmovi se
            # samo citaju iz postojeceg ZIP-a i nikada se ponovo ne generisu.
            import shutil

            PHASE7_CODE_DIR = DRIVE_ROOT / 'phase7_code'
            for module_name in ('decoder.py', 'evaluation.py'):
                drive_source = PHASE7_CODE_DIR / module_name
                repo_target = REPO / 'lipnet' / module_name
                if drive_source.exists():
                    shutil.copy2(drive_source, repo_target)
                    print('Ucitan Faza 7 modul sa Drive-a:', drive_source)
                assert repo_target.exists(), (
                    f'Nedostaje {module_name}. Ocekivan je u checkout-u ili u '
                    f'{PHASE7_CODE_DIR}.'
                )
            importlib.invalidate_caches()
            """),
            md("## Steps\n\n### 1. Učitaj test mouth frejmove i anotacije"),
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
            CORPUS_ROOT = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align')).parents[2]
            """),
            code("""
            from torch.utils.data import DataLoader, Subset

            from data.splits import TEST_SPEAKERS
            from lipnet.dataset import SerbianDataset, variable_length_collate
            from lipnet.train import scan_ctc_compatibility

            raw_test_dataset = SerbianDataset(
                MOUTH_ROOT, CORPUS_ROOT, TEST_SPEAKERS, phase='test'
            )
            ctc_report = scan_ctc_compatibility(raw_test_dataset)
            test_dataset = Subset(raw_test_dataset, ctc_report.valid_indices)
            test_loader = DataLoader(
                test_dataset,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                collate_fn=variable_length_collate,
                pin_memory=True,
            )
            print('Test:', len(test_dataset), 'CTC-odbačeno:', ctc_report.invalid_count)
            """),
            md("### 2. Učitaj tačno provereni Phase 5 checkpoint"),
            code("""
            from lipnet.dataset import SERBIAN_LETTERS
            from lipnet.model import LipNet

            PHASE5_DIR = DRIVE_ROOT / 'phase5_length_aware_v2'
            BEST_CHECKPOINT = PHASE5_DIR / 'best.pt'
            PHASE5_RESULTS = PHASE5_DIR / 'results.json'
            EXPERIMENT_RESULTS = PHASE5_DIR / 'robustness_results.json'
            EXPERIMENT_PLOT = PHASE5_DIR / 'robustness_metrics.png'
            assert BEST_CHECKPOINT.exists() and PHASE5_RESULTS.exists()

            checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu', weights_only=False)
            assert checkpoint['metadata']['training_protocol'] == 'length-aware-bigru-corpus-metrics-v2'
            model = LipNet(num_classes=1 + len(SERBIAN_LETTERS)).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()

            hasher = hashlib.sha256()
            with BEST_CHECKPOINT.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(chunk)
            checkpoint_sha256 = hasher.hexdigest()
            phase5_results = json.loads(PHASE5_RESULTS.read_text(encoding='utf-8'))
            assert phase5_results['best_checkpoint_sha256'] == checkpoint_sha256
            assert phase5_results['metrics_definition'] == 'corpus-edit-distance-v1'

            """),
            md("### 3. Definiši determinističke ulazne varijante"),
            code("""
            def map_frames(video, operation):
                batch, channels, time, height, width = video.shape
                frames = video.permute(0, 2, 1, 3, 4).reshape(
                    batch * time, channels, height, width
                )
                changed = operation(frames)
                return changed.reshape(batch, time, channels, height, width).permute(0, 2, 1, 3, 4)

            def identity(video):
                return video

            def resolution(video, height, width):
                def resize(frames):
                    low = F.interpolate(
                        frames, size=(height, width), mode='bilinear', align_corners=False
                    )
                    return F.interpolate(
                        low, size=(64, 128), mode='bilinear', align_corners=False
                    )
                return map_frames(video, resize)

            def gaussian_blur(video, kernel_size=5, sigma=1.0):
                coordinates = torch.arange(kernel_size, device=video.device) - kernel_size // 2
                kernel_1d = torch.exp(-(coordinates.float() ** 2) / (2 * sigma**2))
                kernel_1d /= kernel_1d.sum()
                kernel_2d = torch.outer(kernel_1d, kernel_1d)
                kernel = kernel_2d.expand(3, 1, kernel_size, kernel_size)
                def blur(frames):
                    padded = F.pad(
                        frames,
                        (kernel_size // 2,) * 4,
                        mode='reflect',
                    )
                    return F.conv2d(padded, kernel, groups=3)
                return map_frames(video, blur)

            def crop_shift(video, dx=4, dy=2):
                shifted = torch.zeros_like(video)
                shifted[:, :, :, dy:, dx:] = video[:, :, :, :-dy, :-dx]
                return shifted

            conditions = {
                'baseline_128x64': identity,
                'resolution_96x48': lambda video: resolution(video, 48, 96),
                'resolution_64x32': lambda video: resolution(video, 32, 64),
                'gaussian_blur_5_sigma1': gaussian_blur,
                'crop_shift_dx4_dy2': crop_shift,
            }
            """),
            md("### 4. Izmeri sve uslove istim dekoderom i definicijom metrika"),
            code("""
            from lipnet.train import greedy_decode, reference_text, sequence_metrics

            condition_outputs = {}
            for name, transform in conditions.items():
                predictions = []
                references = []
                with torch.inference_mode():
                    for batch in test_loader:
                        video = transform(batch['vid'].to(DEVICE, non_blocking=True))
                        logits = model(video, lengths=batch['vid_len'])
                        predictions.extend(
                            greedy_decode(logits, output_lengths=batch['vid_len'])
                        )
                        references.extend(reference_text(batch))
                metrics = sequence_metrics(predictions, references)
                condition_outputs[name] = {
                    'metrics': metrics,
                    'predictions': predictions,
                    'references': references,
                }
                print(name, json.dumps(metrics, ensure_ascii=False))

            baseline_metrics = condition_outputs['baseline_128x64']['metrics']
            for metric in ('wer', 'cer', 'sentence_exact_match'):
                assert math.isclose(
                    baseline_metrics[metric],
                    phase5_results['test'][metric],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ), (metric, baseline_metrics[metric], phase5_results['test'][metric])
            print('PASS: baseline se tačno poklapa sa sačuvanom Phase 5 evaluacijom.')
            """),
            md("## Checks\n\n### 5. Sačuvaj delte, kvalitativne greške i grafikon"),
            code("""
            summary = {}
            for name, output in condition_outputs.items():
                metrics = output['metrics']
                errors = [
                    {'reference': reference, 'prediction': prediction}
                    for reference, prediction in zip(output['references'], output['predictions'])
                    if reference != prediction
                ][:10]
                summary[name] = {
                    **metrics,
                    'wer_delta_vs_baseline': metrics['wer'] - baseline_metrics['wer'],
                    'cer_delta_vs_baseline': metrics['cer'] - baseline_metrics['cer'],
                    'qualitative_errors': errors,
                }

            payload = {
                'phase': 6,
                'checkpoint_sha256': checkpoint_sha256,
                'metrics_definition': 'corpus-edit-distance-v1',
                'test_samples': len(test_dataset),
                'environment': {
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0),
                },
                'conditions': summary,
            }
            EXPERIMENT_RESULTS.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + '\\n',
                encoding='utf-8',
            )

            names = list(summary)
            figure, axes = plt.subplots(1, 2, figsize=(13, 4))
            for axis, metric in zip(axes, ('wer', 'cer')):
                values = [summary[name][metric] for name in names]
                axis.barh(names, values)
                axis.set_xlabel(metric.upper())
                axis.set_xlim(left=0)
                axis.grid(axis='x', alpha=0.25)
                for index, value in enumerate(values):
                    axis.text(value, index, f' {value:.4f}', va='center')
            plt.tight_layout()
            figure.savefig(EXPERIMENT_PLOT, dpi=160, bbox_inches='tight')
            plt.show()
            print('Sačuvano:', EXPERIMENT_RESULTS)
            print('Sačuvano:', EXPERIMENT_PLOT)
            """),
            md("""
            ## Next Steps

            Rezultate tumači tek nakon uspešnog baseline `assert` poređenja. Veći pozitivni
            `wer_delta_vs_baseline` znači da je intervencija pogoršala model. U završni tekst
            prenesi samo vrednosti iz `robustness_results.json` i navedi checkpoint SHA-256,
            jer time ostaje proverljivo koji je model evaluiran.
            """),
        ],
        gpu=True,
    )


def phase7() -> nbf.NotebookNode:
    return notebook(
        "Faza 7 — CTC beam search i karakterni jezički model",
        [
            md("""
            ## Goal

            Uporedi postojeći greedy CTC baseline sa prefix beam search dekoderom bez
            jezičkog modela i sa karakternim 5-gram modelom. LipNet checkpoint, test split
            i mouth frejmovi ostaju potpuno isti; ne radi se novi trening niti BlazeFace
            preprocessing.
            """),
            md("""
            ## Setup

            Ovaj notebook koristi postojeći `ai_speak_lip.zip`, originalne `.align`
            anotacije i `phase5_length_aware_v2/best.pt`. GPU je potreban samo za jedno
            keširanje validation/test logit-a. Sve decoder konfiguracije zatim rade nad
            istim CPU kešom. Karakterni LM se fit-uje isključivo nad train transkriptima;
            validation bira hiperparametre, a test se koristi jednom za zaključane dekodere.
            """),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10'],
                check=True,
            )
            """),
            code(repo_setup_cell(refresh_imports=True)),
            code("""
            import hashlib
            import inspect
            import json
            import math
            import time
            import zipfile
            from dataclasses import asdict

            import matplotlib.pyplot as plt
            import torch

            from lipnet.model import LipNet

            assert 'lengths' in inspect.signature(LipNet.forward).parameters
            assert torch.cuda.is_available(), 'Uključi T4/L4 GPU za jedno keširanje logit-a.'
            DEVICE = torch.device('cuda')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print('LipNet:', inspect.getfile(LipNet), inspect.signature(LipNet.forward))
            print('GPU:', torch.cuda.get_device_name(0))
            """),
            code(drive_setup_cell()),
            md("## Steps\n\n### 1. Učitaj postojeće frejmove i anotacije — bez preprocessinga"),
            code("""
            MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            SOURCE_ARCHIVE = Path('/content/drive/MyDrive/processed.zip')  # promeni samo putanju
            MOUTH_ROOT = Path('/content/ai_speak_lip')
            ALIGN_EXTRACT = Path('/content/ai_speak_align')
            assert MOUTH_ARCHIVE.exists(), MOUTH_ARCHIVE
            assert SOURCE_ARCHIVE.exists(), SOURCE_ARCHIVE

            # Samo raspakivanje već napravljenih artefakata. Nema face detectora/BlazeFace-a.
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
            CORPUS_ROOT = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align')).parents[2]
            print('Koristim zamrznuti mouth artefakt:', MOUTH_ARCHIVE)
            """),
            code("""
            from torch.utils.data import DataLoader, Subset

            from data.splits import SPLITS
            from lipnet.dataset import SerbianDataset, variable_length_collate
            from lipnet.train import scan_ctc_compatibility

            raw_datasets = {
                name: SerbianDataset(MOUTH_ROOT, CORPUS_ROOT, speakers, phase='test')
                for name, speakers in SPLITS.items()
            }
            reports = {
                name: scan_ctc_compatibility(dataset)
                for name, dataset in raw_datasets.items()
            }
            datasets = {
                name: Subset(raw_datasets[name], report.valid_indices)
                for name, report in reports.items()
            }

            def make_loader(name):
                return DataLoader(
                    datasets[name], batch_size=2, shuffle=False, num_workers=2,
                    collate_fn=variable_length_collate, pin_memory=True,
                )

            train_token_sequences = [
                raw_datasets['train'].ctc_lengths(index)[1].tolist()
                for index in reports['train'].valid_indices
            ]
            for name in ('train', 'validation', 'test'):
                print(name, 'valid=', len(datasets[name]), 'odbačeno=', reports[name].invalid_count)
            """),
            md("### 2. Proveri checkpoint i jednom keširaj validation/test logit-e"),
            code("""
            from lipnet.dataset import SERBIAN_LETTERS
            from lipnet.model import LipNet
            from lipnet.train import reference_text

            PHASE5_DIR = DRIVE_ROOT / 'phase5_length_aware_v2'
            BEST_CHECKPOINT = PHASE5_DIR / 'best.pt'
            PHASE5_RESULTS = PHASE5_DIR / 'results.json'
            EMISSIONS_CACHE = PHASE5_DIR / 'decoder_emissions_v1.pt'
            DECODER_RESULTS = PHASE5_DIR / 'decoder_results_v1.json'
            DECODER_PREDICTIONS = PHASE5_DIR / 'decoder_predictions_v1.json'
            DECODER_PLOT = PHASE5_DIR / 'decoder_metrics_v1.png'
            DECODER_TUNING = PHASE5_DIR / 'decoder_validation_v1.json'
            assert BEST_CHECKPOINT.exists() and PHASE5_RESULTS.exists()

            checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu', weights_only=False)
            assert checkpoint['metadata']['training_protocol'] == 'length-aware-bigru-corpus-metrics-v2'
            model = LipNet(num_classes=1 + len(SERBIAN_LETTERS)).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()

            hasher = hashlib.sha256()
            with BEST_CHECKPOINT.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(chunk)
            checkpoint_sha256 = hasher.hexdigest()
            phase5_results = json.loads(PHASE5_RESULTS.read_text(encoding='utf-8'))
            assert phase5_results['best_checkpoint_sha256'] == checkpoint_sha256
            assert phase5_results['metrics_definition'] == 'corpus-edit-distance-v1'

            split_signatures = {}
            for name in ('validation', 'test'):
                identities = [
                    f'{raw_datasets[name].data[index][1]}/{raw_datasets[name].data[index][2]}'
                    for index in reports[name].valid_indices
                ]
                split_signatures[name] = hashlib.sha256(
                    json.dumps(identities, ensure_ascii=False).encode('utf-8')
                ).hexdigest()

            def collect_emissions(name):
                emissions, references = [], []
                with torch.inference_mode():
                    for batch in make_loader(name):
                        logits = model(
                            batch['vid'].to(DEVICE, non_blocking=True),
                            lengths=batch['vid_len'],
                        ).cpu()
                        for row, length in zip(logits, batch['vid_len']):
                            emissions.append(row[:int(length)].contiguous())
                        references.extend(reference_text(batch))
                return {'logits': emissions, 'references': references}

            cache = None
            if EMISSIONS_CACHE.exists():
                candidate = torch.load(EMISSIONS_CACHE, map_location='cpu', weights_only=False)
                if (
                    candidate.get('schema_version') == 1
                    and candidate.get('checkpoint_sha256') == checkpoint_sha256
                    and candidate.get('split_signatures') == split_signatures
                    and set(candidate.get('splits', {})) == {'validation', 'test'}
                ):
                    cache = candidate
                    print('Koristim postojeći emissions cache za isti checkpoint.')
            if cache is None:
                cache = {
                    'schema_version': 1,
                    'checkpoint_sha256': checkpoint_sha256,
                    'split_signatures': split_signatures,
                    'splits': {
                        name: collect_emissions(name)
                        for name in ('validation', 'test')
                    },
                }
                torch.save(cache, EMISSIONS_CACHE)
                print('Sačuvano:', EMISSIONS_CACHE)
            for name in ('validation', 'test'):
                assert len(cache['splits'][name]['logits']) == len(datasets[name])
                assert len(cache['splits'][name]['references']) == len(datasets[name])
            del model
            torch.cuda.empty_cache()
            """),
            md("### 3. Fituj 5-gram LM samo nad train transkriptima"),
            code("""
            from lipnet.decoder import BeamSearchConfig, CharacterNGramLM, prefix_beam_decode
            from lipnet.evaluation import paired_bootstrap_delta, slot_error_analysis
            from lipnet.train import greedy_decode, sequence_metrics

            language_model = CharacterNGramLM(
                order=5,
                smoothing=0.1,
                vocabulary=range(1, 1 + len(SERBIAN_LETTERS)),
            ).fit(train_token_sequences)
            assert language_model.training_sequences == len(datasets['train'])
            print('LM training transkripti:', language_model.training_sequences)
            print('Validation/test transkripti nisu korišćeni za fit.')

            def decode_cached(split_name, config=None, use_lm=False):
                split = cache['splits'][split_name]
                started = time.perf_counter()
                predictions = []
                for index, logits in enumerate(split['logits'], start=1):
                    if config is None:
                        predictions.extend(greedy_decode(logits.unsqueeze(0)))
                    else:
                        predictions.extend(prefix_beam_decode(
                            logits.unsqueeze(0),
                            config=config,
                            language_model=language_model if use_lm else None,
                        ))
                    if index % 100 == 0 or index == len(split['logits']):
                        print(split_name, index, '/', len(split['logits']), end='\\r')
                print()
                return {
                    'predictions': predictions,
                    'references': split['references'],
                    'metrics': sequence_metrics(predictions, split['references']),
                    'seconds': time.perf_counter() - started,
                }
            """),
            md("### 4. Validation bira po jedan beam bez LM-a i beam + LM"),
            code("""
            TOKEN_TOPK = 8
            probe_full = BeamSearchConfig(beam_width=25)
            probe_pruned = BeamSearchConfig(beam_width=25, token_topk=TOKEN_TOPK)
            probe_logits = cache['splits']['validation']['logits'][:25]
            probe_full_predictions = [
                prefix_beam_decode(logits.unsqueeze(0), config=probe_full)[0]
                for logits in probe_logits
            ]
            probe_pruned_predictions = [
                prefix_beam_decode(logits.unsqueeze(0), config=probe_pruned)[0]
                for logits in probe_logits
            ]
            pruning_probe_agreement = sum(
                left == right
                for left, right in zip(probe_full_predictions, probe_pruned_predictions)
            ) / len(probe_logits)
            assert pruning_probe_agreement >= 0.95, (
                'Top-k pruning se nedovoljno slaže sa punim beam-om; povećaj TOKEN_TOPK.',
                pruning_probe_agreement,
            )
            print('Top-k/full beam probe agreement:', pruning_probe_agreement)

            tuning_payload = {
                'schema_version': 1,
                'checkpoint_sha256': checkpoint_sha256,
                'validation_split_signature': split_signatures['validation'],
                'lm': {'order': 5, 'smoothing': 0.1, 'training_split': 'train_only'},
                'runs': {},
            }
            if DECODER_TUNING.exists():
                candidate = json.loads(DECODER_TUNING.read_text(encoding='utf-8'))
                expected_header = {key: tuning_payload[key] for key in tuning_payload if key != 'runs'}
                candidate_header = {key: candidate.get(key) for key in expected_header}
                if candidate_header == expected_header:
                    tuning_payload = candidate
                    print('Nastavljam validation tuning:', len(tuning_payload['runs']), 'run-ova')

            def validation_run(name, family, config=None, use_lm=False):
                serialized_config = asdict(config) if config is not None else None
                cached = tuning_payload['runs'].get(name)
                if cached is not None and cached.get('config') == serialized_config:
                    print(name, '— učitano iz validation cache-a', cached['metrics'])
                    return cached
                output = decode_cached('validation', config=config, use_lm=use_lm)
                run = {
                    'family': family,
                    'config': serialized_config,
                    'metrics': output['metrics'],
                    'seconds': output['seconds'],
                }
                tuning_payload['runs'][name] = run
                DECODER_TUNING.write_text(
                    json.dumps(tuning_payload, indent=2, ensure_ascii=False) + '\\n',
                    encoding='utf-8',
                )
                print(name, run['metrics'])
                return run

            validation_runs = {
                'greedy': validation_run('greedy', 'greedy'),
            }

            for width in (10, 25, 50):
                name = f'beam_w{width}_nolm'
                config = BeamSearchConfig(beam_width=width, token_topk=TOKEN_TOPK)
                validation_runs[name] = validation_run(name, 'beam_nolm', config=config)

            # Originalni rad koristi width=200, alpha=1, beta=1.5 na engleskom GRID-u.
            # Za srpski AI-SPEAK alpha/beta se pošteno biraju samo na validation splitu.
            for lm_weight in (0.25, 0.5, 1.0):
                for word_bonus in (0.0, 0.5, 1.5):
                    name = f'beam_w50_lm_a{lm_weight}_b{word_bonus}'
                    config = BeamSearchConfig(
                        beam_width=50,
                        lm_weight=lm_weight,
                        word_bonus=word_bonus,
                        token_topk=TOKEN_TOPK,
                    )
                    validation_runs[name] = validation_run(
                        name, 'beam_lm', config=config, use_lm=True
                    )

            def selection_key(item):
                metrics = item[1]['metrics']
                return (metrics['wer'], metrics['cer'], -metrics['sentence_exact_match'])

            best_nolm_name, best_nolm = min(
                ((name, run) for name, run in validation_runs.items()
                 if run['family'] == 'beam_nolm'),
                key=selection_key,
            )
            best_lm_name, best_lm = min(
                ((name, run) for name, run in validation_runs.items()
                 if run['family'] == 'beam_lm'),
                key=selection_key,
            )
            print('Izabran bez LM:', best_nolm_name, best_nolm['metrics'])
            print('Izabran sa LM:', best_lm_name, best_lm['metrics'])
            """),
            md("## Checks\n\n### 5. Zaključane konfiguracije evaluiraj jednom na testu"),
            code("""
            test_runs = {
                'greedy': {
                    'family': 'greedy',
                    'config': None,
                    **decode_cached('test'),
                },
                best_nolm_name: {
                    'family': 'beam_nolm',
                    'config': best_nolm['config'],
                    **decode_cached(
                        'test', config=BeamSearchConfig(**best_nolm['config'])
                    ),
                },
                best_lm_name: {
                    'family': 'beam_lm',
                    'config': best_lm['config'],
                    **decode_cached(
                        'test', config=BeamSearchConfig(**best_lm['config']), use_lm=True
                    ),
                },
            }

            for metric in ('wer', 'cer', 'sentence_exact_match'):
                assert math.isclose(
                    test_runs['greedy']['metrics'][metric],
                    phase5_results['test'][metric],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ), (metric, test_runs['greedy']['metrics'][metric], phase5_results['test'][metric])
            print('PASS: greedy test se tačno poklapa sa Phase 5 rezultatom.')
            for name, run in test_runs.items():
                print(name, json.dumps(run['metrics'], ensure_ascii=False), f"{run['seconds']:.1f}s")
            """),
            md("### 6. Paired bootstrap, analiza pozicija i trajni artefakti"),
            code("""
            SLOT_NAMES = ('komanda', 'slovo_1', 'smer', 'slovo_2', 'dan', 'broj')
            baseline = test_runs['greedy']
            comparisons = {}
            for name, run in test_runs.items():
                if name == 'greedy':
                    continue
                comparisons[name] = {
                    'paired_bootstrap_vs_greedy': paired_bootstrap_delta(
                        baseline['predictions'], run['predictions'], run['references'],
                        iterations=2000, confidence=0.95, seed=0,
                    ),
                    'slot_analysis': slot_error_analysis(
                        run['predictions'], run['references'], slot_names=SLOT_NAMES,
                    ),
                }

            def compact_run(run):
                return {
                    'family': run['family'],
                    'config': run['config'],
                    'metrics': run['metrics'],
                    'seconds': run['seconds'],
                }

            payload = {
                'phase': 7,
                'protocol': 'ctc-prefix-beam-char5gram-validation-selected-v1',
                'checkpoint_sha256': checkpoint_sha256,
                'metrics_definition': 'corpus-edit-distance-v1',
                'lm': {
                    'type': 'character_backoff_add_k',
                    'order': language_model.order,
                    'smoothing': language_model.smoothing,
                    'training_split': 'train_only',
                    'training_sequences': language_model.training_sequences,
                },
                'selection_metric': 'validation_wer_then_cer_then_exact',
                'token_pruning_probe': {
                    'samples': len(probe_logits),
                    'token_topk': TOKEN_TOPK,
                    'agreement_with_full_beam': pruning_probe_agreement,
                },
                'validation_runs': {
                    name: compact_run(run) for name, run in validation_runs.items()
                },
                'selected_test_runs': {
                    name: compact_run(run) for name, run in test_runs.items()
                },
                'comparisons': comparisons,
                'environment': {
                    'torch_version': torch.__version__,
                    'cuda_version': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0),
                },
            }
            DECODER_RESULTS.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + '\\n', encoding='utf-8'
            )
            prediction_payload = {
                'phase': 7,
                'checkpoint_sha256': checkpoint_sha256,
                'references': baseline['references'],
                'predictions': {
                    name: run['predictions'] for name, run in test_runs.items()
                },
            }
            DECODER_PREDICTIONS.write_text(
                json.dumps(prediction_payload, indent=2, ensure_ascii=False) + '\\n',
                encoding='utf-8',
            )

            names = list(test_runs)
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            for axis, metric in zip(axes, ('wer', 'cer')):
                values = [test_runs[name]['metrics'][metric] for name in names]
                axis.barh(names, values)
                axis.set_xlabel(metric.upper())
                axis.grid(axis='x', alpha=0.25)
                for index, value in enumerate(values):
                    axis.text(value, index, f' {value:.4f}', va='center')
            plt.tight_layout()
            figure.savefig(DECODER_PLOT, dpi=160, bbox_inches='tight')
            plt.show()
            print('Sačuvano:', DECODER_RESULTS)
            print('Sačuvano:', DECODER_PREDICTIONS)
            print('Sačuvano:', DECODER_PLOT)
            """),
            md("""
            ## Next Steps

            Faza 7 je završena samo ako greedy `assert` prođe, LM navodi `train_only`,
            validation izbor je zabeležen i `decoder_results_v1.json` sadrži paired bootstrap
            intervale. Ovi rezultati se zatim prenose u repo i koriste za konsolidovani
            notebook za odbranu; finalni izveštaj se piše odvojeno, po naknadno dogovorenom
            formatu.
            """),
        ],
        gpu=True,
    )


def phase8() -> nbf.NotebookNode:
    return notebook(
        "Faza 8 — konsolidovani notebook za odbranu",
        [
            md("""
            ## Ukratko

            Ovaj notebook objedinjuje potvrđeni tok vizuelnog prepoznavanja srpskog
            govora: podatke, model, baseline, robustnost, CTC dekodiranje i analizu
            grešaka. Ne pokreće novi trening niti preprocessing. Finalni test rezultat
            beam+5-gram dekodera je **WER 41,20%** i **CER 14,70%** nad 540 govornik-
            disjoint primera.
            """),
            md("""
            ## Kontekst i metod

            - Zamrznuti mouth frejmovi dolaze iz `ai_speak_lip.zip`.
            - Model koristi zaključani Phase 5 checkpoint.
            - Rezultati Faza 3–7 čitaju se iz verzionisanih JSON artefakata.
            - GPU se koristi samo za demonstraciju jedne predikcije.
            - Train-only 5-gram LM se ponovo fituje bez validation/test transkripata.
            - Sve figure se prikazuju inline i čuvaju u `MyDrive/LipNet/phase8_report/`.

            Matrica fonema/vizema iz teorijske prezentacije pripada originalnom LipNet
            radu na GRID-u. Matrice u ovom notebooku računaju se isključivo iz naših
            AI-SPEAK test predikcija.
            """),
            md("## Setup\n\n### 1. Učitaj repozitorijum i zavisnosti"),
            code("""
            import subprocess, sys
            subprocess.run(
                [sys.executable, '-m', 'pip', 'install', '-q',
                 'editdistance>=0.8.1', 'opencv-python-headless>=4.10'],
                check=True,
            )
            """),
            code(repo_setup_cell(refresh_imports=True)),
            code("""
            import hashlib
            import json
            import math
            import shutil
            import zipfile
            from pathlib import Path

            import editdistance
            import matplotlib.pyplot as plt
            from matplotlib.ticker import PercentFormatter
            import numpy as np
            import pandas as pd
            import torch
            from IPython.display import display

            assert torch.cuda.is_available(), 'Uključi GPU runtime u Colab-u.'
            DEVICE = torch.device('cuda')
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            print('GPU:', torch.cuda.get_device_name(0))
            """),
            code(drive_setup_cell(force_remount=True)),
            code("""
            REPORT_DIR = DRIVE_ROOT / 'phase8_report'
            REPORT_DIR.mkdir(parents=True, exist_ok=True)

            COLORS = {
                'ink': '#1B2A36',
                'blue': '#15728E',
                'orange': '#E58A2B',
                'gold': '#D4A72C',
                'olive': '#6B7D3D',
                'pink': '#B45A7C',
                'grid': '#D9E0E5',
            }
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': COLORS['ink'],
                'axes.labelcolor': COLORS['ink'],
                'text.color': COLORS['ink'],
                'xtick.color': COLORS['ink'],
                'ytick.color': COLORS['ink'],
                'font.size': 11,
            })

            def finish_figure(figure, filename):
                destination = REPORT_DIR / filename
                figure.savefig(destination, dpi=180, bbox_inches='tight', facecolor='white')
                plt.show()
                plt.close(figure)
                print('Sačuvano:', destination)

            def chart_header(axis, title, subtitle):
                axis.set_title(title, loc='left', weight='bold', pad=30)
                axis.text(
                    0, 1.015, subtitle, transform=axis.transAxes,
                    color='#5B6B78', va='bottom',
                )
            """),
            md("## Podaci i integritet\n\n### 2. Učitaj potvrđene artefakte Faza 3–7"),
            code("""
            RESULTS_DIR = REPO / 'docs' / 'results'

            def read_json(name):
                path = RESULTS_DIR / name
                assert path.exists(), path
                return json.loads(path.read_text(encoding='utf-8'))

            phase3 = read_json('phase3_dataset_audit.json')
            phase4 = read_json('phase4_transfer_ctc_audit.json')
            phase5 = read_json('phase5_results.json')
            phase6 = read_json('phase6_robustness_results.json')
            phase7 = read_json('decoder_results_v1.json')
            decoder_predictions = read_json('decoder_predictions_v1.json')

            CHECKPOINT_SHA256 = phase5['best_checkpoint_sha256']
            assert phase4['num_classes'] == 29
            assert phase6['checkpoint_sha256'] == CHECKPOINT_SHA256
            assert phase7['checkpoint_sha256'] == CHECKPOINT_SHA256
            assert decoder_predictions['checkpoint_sha256'] == CHECKPOINT_SHA256
            assert phase7['lm']['training_split'] == 'train_only'
            assert phase7['lm']['training_sequences'] == phase3['samples_by_split']['train']
            assert len(decoder_predictions['references']) == phase5['test_samples'] == 540
            assert all(
                len(values) == 540
                for values in decoder_predictions['predictions'].values()
            )
            print('PASS: artefakti Faza 3–7 su kompletni i koriste isti checkpoint.')
            """),
            md("### 3. Prikaži speaker-disjoint podelu"),
            code("""
            split_order = ('train', 'validation', 'test')
            split_labels = ('Trening', 'Validacija', 'Test')
            split_values = [phase3['samples_by_split'][name] for name in split_order]

            figure, axis = plt.subplots(figsize=(8, 4.5))
            bars = axis.bar(split_labels, split_values, color=COLORS['blue'], edgecolor=COLORS['ink'])
            chart_header(
                axis,
                'Speaker-disjoint podela AI-SPEAK skupa',
                'Broj primera po skupu; govornici se ne preklapaju',
            )
            axis.set_ylabel('Broj primera')
            axis.set_ylim(0, max(split_values) * 1.18)
            axis.grid(axis='y', color=COLORS['grid'], alpha=0.7)
            axis.spines[['top', 'right']].set_visible(False)
            axis.bar_label(bars, labels=[f'{value:,}'.replace(',', '.') for value in split_values],
                           padding=4, weight='bold')
            finish_figure(figure, '01_dataset_split.png')
            """),
            md("""
            ## Obavezna GPU demonstracija

            ### 4. Učitaj zamrznute frejmove i anotacije

            Raspakuju se samo postojeći artefakti. Detekcija lica i preprocessing se
            ne pokreću ponovo. ZIP arhive se prvo kopiraju na lokalni Colab disk da
            raspakivanje ne zavisi od nestabilnog nasumičnog čitanja Drive mount-a.
            """),
            code("""
            MOUTH_ARCHIVE = DRIVE_ROOT / 'ai_speak_lip.zip'
            SOURCE_ARCHIVE = Path('/content/drive/MyDrive/processed.zip')
            LOCAL_MOUTH_ARCHIVE = Path('/content/ai_speak_lip.zip')
            LOCAL_SOURCE_ARCHIVE = Path('/content/processed.zip')
            MOUTH_ROOT = Path('/content/ai_speak_lip')
            ALIGN_EXTRACT = Path('/content/ai_speak_align')
            assert MOUTH_ARCHIVE.exists(), MOUTH_ARCHIVE
            assert SOURCE_ARCHIVE.exists(), SOURCE_ARCHIVE

            def stage_drive_archive(source, destination):
                try:
                    source_size = source.stat().st_size
                    if not destination.exists() or destination.stat().st_size != source_size:
                        print(f'Kopiram {source.name} na lokalni Colab disk...')
                        shutil.copyfile(source, destination)
                    assert destination.stat().st_size == source_size
                    assert zipfile.is_zipfile(destination), f'Neispravan ZIP: {destination}'
                    return destination
                except OSError as error:
                    if getattr(error, 'errno', None) == 107:
                        raise RuntimeError(
                            'Google Drive veza je prekinuta. Ponovo pokreni Drive mount '
                            'ćeliju iznad, pa zatim ponovi ovu ćeliju.'
                        ) from error
                    raise

            if not next(MOUTH_ROOT.glob('spk*/video/video_a/*'), None):
                MOUTH_ROOT.mkdir(parents=True, exist_ok=True)
                local_archive = stage_drive_archive(MOUTH_ARCHIVE, LOCAL_MOUTH_ARCHIVE)
                with zipfile.ZipFile(local_archive) as archive:
                    archive.extractall(MOUTH_ROOT)
                local_archive.unlink()
            if not next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align'), None):
                ALIGN_EXTRACT.mkdir(parents=True, exist_ok=True)
                local_archive = stage_drive_archive(SOURCE_ARCHIVE, LOCAL_SOURCE_ARCHIVE)
                with zipfile.ZipFile(local_archive) as archive:
                    members = [
                        member for member in archive.infolist()
                        if '/alignment/' in f'/{member.filename}'
                        and member.filename.endswith('.align')
                    ]
                    archive.extractall(ALIGN_EXTRACT, members=members)
                local_archive.unlink()
            CORPUS_ROOT = next(ALIGN_EXTRACT.rglob('spk*/alignment/*.align')).parents[2]
            print('Mouth frejmovi:', MOUTH_ROOT)
            print('Anotacije:', CORPUS_ROOT)
            """),
            code("""
            from data.splits import SPLITS
            from lipnet.dataset import SERBIAN_LETTERS, SerbianDataset, variable_length_collate
            from lipnet.train import scan_ctc_compatibility

            raw_datasets = {
                name: SerbianDataset(MOUTH_ROOT, CORPUS_ROOT, speakers, phase='test')
                for name, speakers in SPLITS.items()
            }
            reports = {
                name: scan_ctc_compatibility(dataset)
                for name, dataset in raw_datasets.items()
            }
            for name in ('train', 'validation', 'test'):
                assert reports[name].invalid_count == 0
                assert len(reports[name].valid_indices) == phase3['samples_by_split'][name]
                print(name, 'valid=', len(reports[name].valid_indices))
            """),
            md("### 5. Učitaj checkpoint, fituj train-only LM i dekodiraj primer 42"),
            code("""
            from lipnet.decoder import BeamSearchConfig, CharacterNGramLM, prefix_beam_decode
            from lipnet.model import LipNet
            from lipnet.train import greedy_decode

            PHASE5_DIR = DRIVE_ROOT / 'phase5_length_aware_v2'
            BEST_CHECKPOINT = PHASE5_DIR / 'best.pt'
            assert BEST_CHECKPOINT.exists(), BEST_CHECKPOINT

            hasher = hashlib.sha256()
            with BEST_CHECKPOINT.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    hasher.update(chunk)
            assert hasher.hexdigest() == CHECKPOINT_SHA256

            checkpoint = torch.load(BEST_CHECKPOINT, map_location='cpu', weights_only=False)
            assert checkpoint['metadata']['training_protocol'] == 'length-aware-bigru-corpus-metrics-v2'
            model = LipNet(num_classes=1 + len(SERBIAN_LETTERS)).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()

            train_token_sequences = [
                raw_datasets['train'].ctc_lengths(index)[1].tolist()
                for index in reports['train'].valid_indices
            ]
            language_model = CharacterNGramLM(
                order=phase7['lm']['order'],
                smoothing=phase7['lm']['smoothing'],
                vocabulary=range(1, 1 + len(SERBIAN_LETTERS)),
            ).fit(train_token_sequences)
            assert language_model.training_sequences == 2877

            DEMO_INDEX = 42
            raw_index = reports['test'].valid_indices[DEMO_INDEX]
            sample = raw_datasets['test'][raw_index]
            batch = variable_length_collate([sample])
            with torch.inference_mode():
                logits = model(batch['vid'].to(DEVICE), batch['vid_len'])

            reference = SerbianDataset.arr2txt(sample['txt'].tolist())
            greedy_prediction = greedy_decode(logits, output_lengths=batch['vid_len'])[0]
            lm_config = phase7['selected_test_runs']['beam_w50_lm_a1.0_b0.5']['config']
            lm_prediction = prefix_beam_decode(
                logits,
                config=BeamSearchConfig(**lm_config),
                language_model=language_model,
                output_lengths=batch['vid_len'],
            )[0]

            assert reference == decoder_predictions['references'][DEMO_INDEX]
            assert greedy_prediction == decoder_predictions['predictions']['greedy'][DEMO_INDEX]
            assert lm_prediction == decoder_predictions['predictions']['beam_w50_lm_a1.0_b0.5'][DEMO_INDEX]
            assert lm_prediction == reference
            print('Referenca:', reference)
            print('Greedy:   ', greedy_prediction)
            print('Beam + LM:', lm_prediction)
            """),
            code("""
            frame_count = int(sample['vid_len'])
            frame_indices = np.linspace(0, frame_count - 1, 6, dtype=int)
            figure, axes = plt.subplots(1, len(frame_indices), figsize=(13, 2.7))
            for axis, frame_index in zip(axes, frame_indices):
                frame = sample['vid'][:, frame_index].permute(1, 2, 0).numpy()[..., ::-1]
                axis.imshow(np.clip(frame, 0, 1))
                axis.set_title(f'frejm {frame_index}')
                axis.axis('off')
            figure.suptitle(
                'Region usana kroz demonstracionu sekvencu',
                x=0.02, y=0.98, ha='left', weight='bold',
            )
            figure.text(
                0.02, 0.84, f'Referenca: {reference}',
                color='#5B6B78', ha='left',
            )
            figure.subplots_adjust(left=0.02, right=0.99, top=0.72, bottom=0.08, wspace=0.16)
            finish_figure(figure, '02_gpu_demo_mouth_frames.png')
            """),
            md("## Rezultati\n\n### 6. Robustnost ulaza"),
            code("""
            condition_order = (
                'baseline_128x64', 'resolution_96x48', 'resolution_64x32',
                'gaussian_blur_5_sigma1', 'crop_shift_dx4_dy2',
            )
            condition_labels = ('Osnovni ulaz\\n128×64', '96×48', '64×32', 'Zamućenje', 'Pomeranje\\nROI-ja')
            wer_values = [phase6['conditions'][name]['wer'] for name in condition_order]
            cer_values = [phase6['conditions'][name]['cer'] for name in condition_order]
            positions = np.arange(len(condition_order))
            width = 0.36

            figure, axis = plt.subplots(figsize=(10, 5.2))
            wer_bars = axis.bar(positions - width / 2, wer_values, width, label='WER',
                                color=COLORS['blue'], edgecolor=COLORS['ink'])
            cer_bars = axis.bar(positions + width / 2, cer_values, width, label='CER',
                                color=COLORS['orange'], edgecolor=COLORS['ink'])
            chart_header(
                axis,
                'Robustnost modela na promene ulaza',
                'Corpus-level metrike nad istih 540 test primera',
            )
            axis.set_xticks(positions, condition_labels)
            axis.set_ylim(0, 0.52)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis='y', color=COLORS['grid'], alpha=0.7)
            axis.spines[['top', 'right']].set_visible(False)
            axis.legend(frameon=False, ncol=2, loc='upper right')
            axis.bar_label(wer_bars, labels=[f'{value:.1%}' for value in wer_values], padding=3, fontsize=9)
            axis.bar_label(cer_bars, labels=[f'{value:.1%}' for value in cer_values], padding=3, fontsize=9)
            finish_figure(figure, '03_robustness.png')
            """),
            md("### 7. Greedy, beam i beam+LM dekodiranje"),
            code("""
            decoder_order = ('greedy', 'beam_w50_nolm', 'beam_w50_lm_a1.0_b0.5')
            decoder_labels = ('Greedy', 'Beam bez LM-a', 'Beam + 5-gram LM')
            selected_runs = phase7['selected_test_runs']
            decoder_wer = [selected_runs[name]['metrics']['wer'] for name in decoder_order]
            decoder_cer = [selected_runs[name]['metrics']['cer'] for name in decoder_order]

            figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1.05, 1]})
            positions = np.arange(len(decoder_order))
            width = 0.36
            wer_bars = axes[0].bar(positions - width / 2, decoder_wer, width, label='WER',
                                   color=COLORS['blue'], edgecolor=COLORS['ink'])
            cer_bars = axes[0].bar(positions + width / 2, decoder_cer, width, label='CER',
                                   color=COLORS['orange'], edgecolor=COLORS['ink'])
            axes[0].set_title('Apsolutne test metrike', loc='left', weight='bold')
            axes[0].set_xticks(positions, decoder_labels)
            axes[0].set_ylim(0, 0.52)
            axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
            axes[0].grid(axis='y', color=COLORS['grid'], alpha=0.7)
            axes[0].spines[['top', 'right']].set_visible(False)
            axes[0].legend(frameon=False, ncol=2)
            axes[0].bar_label(wer_bars, labels=[f'{value:.1%}' for value in decoder_wer], padding=3, fontsize=9)
            axes[0].bar_label(cer_bars, labels=[f'{value:.1%}' for value in decoder_cer], padding=3, fontsize=9)

            comparison_specs = (
                ('beam_w50_nolm', 'wer', 'Beam bez LM-a — WER'),
                ('beam_w50_nolm', 'cer', 'Beam bez LM-a — CER'),
                ('beam_w50_lm_a1.0_b0.5', 'wer', 'Beam + LM — WER'),
                ('beam_w50_lm_a1.0_b0.5', 'cer', 'Beam + LM — CER'),
            )
            delta_values, delta_low, delta_high, delta_labels = [], [], [], []
            for comparison_name, metric, label in comparison_specs:
                bootstrap = phase7['comparisons'][comparison_name]['paired_bootstrap_vs_greedy']
                value = 100 * bootstrap[f'{metric}_delta']
                low = 100 * bootstrap[f'{metric}_delta_ci_low']
                high = 100 * bootstrap[f'{metric}_delta_ci_high']
                delta_values.append(value)
                delta_low.append(value - low)
                delta_high.append(high - value)
                delta_labels.append(label)
            y_positions = np.arange(len(delta_labels))
            axes[1].errorbar(
                delta_values, y_positions, xerr=[delta_low, delta_high], fmt='o',
                color=COLORS['pink'], ecolor=COLORS['ink'], capsize=4, markersize=7,
            )
            axes[1].axvline(0, color=COLORS['ink'], linewidth=1)
            axes[1].set_yticks(y_positions, delta_labels)
            axes[1].invert_yaxis()
            axes[1].set_xlabel('Promena prema greedy baseline-u (procentni poeni)')
            axes[1].set_title('Upareni bootstrap — 95% interval', loc='left', weight='bold')
            axes[1].grid(axis='x', color=COLORS['grid'], alpha=0.7)
            axes[1].spines[['top', 'right']].set_visible(False)
            for value, y_position in zip(delta_values, y_positions):
                axes[1].text(value, y_position - 0.18, f'{value:+.2f} p.p.', ha='center', fontsize=9)
            axes[1].set_xlim(-5.05, 0.25)

            figure.suptitle(
                'Uticaj CTC dekodera na test rezultat',
                x=0.01, y=0.99, ha='left', weight='bold',
            )
            figure.text(0.01, 0.01, 'N = 540; negativna delta označava poboljšanje WER-a/CER-a',
                        color='#5B6B78')
            figure.tight_layout(rect=(0, 0.06, 1, 0.92))
            finish_figure(figure, '04_decoder_comparison.png')
            """),
            md("## Analiza grešaka\n\n### 8. Uporedi tačnost po šest pozicija rečenice"),
            code("""
            from lipnet.evaluation import slot_confusion_analysis, slot_error_analysis

            SLOT_NAMES = ('komanda', 'slovo_1', 'smer', 'slovo_2', 'dan', 'broj')
            references = decoder_predictions['references']
            predictions_by_decoder = decoder_predictions['predictions']
            slot_analyses = {
                name: slot_error_analysis(predictions_by_decoder[name], references, slot_names=SLOT_NAMES)
                for name in decoder_order
            }

            figure, axis = plt.subplots(figsize=(11, 5.5))
            positions = np.arange(len(SLOT_NAMES))
            width = 0.25
            for offset, name, label, color in zip(
                (-width, 0, width), decoder_order, decoder_labels,
                (COLORS['blue'], COLORS['gold'], COLORS['pink']),
            ):
                values = [slot_analyses[name]['slots'][slot]['accuracy'] for slot in SLOT_NAMES]
                axis.bar(positions + offset, values, width, label=label, color=color, edgecolor=COLORS['ink'])
            chart_header(
                axis,
                'Tačnost dekodera po poziciji AI-SPEAK rečenice',
                'Poravnata tačnost nad 540 test primera',
            )
            axis.set_xticks(positions, ('Komanda', 'Slovo 1', 'Smer', 'Slovo 2', 'Dan', 'Broj'))
            axis.set_ylim(0, 1.0)
            axis.yaxis.set_major_formatter(PercentFormatter(1.0))
            axis.grid(axis='y', color=COLORS['grid'], alpha=0.7)
            axis.spines[['top', 'right']].set_visible(False)
            axis.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.99))
            finish_figure(figure, '05_slot_accuracy.png')
            """),
            md("### 9. Matrice konfuzije finalnog beam+LM dekodera"),
            code("""
            reference_tokens = [text.split() for text in references]
            slot_vocabularies = {
                'komanda': ('obriši', 'dalje', 'kraj', 'pošalji', 'odustani', 'početak', 'potvrdi'),
                'slovo_1': tuple(sorted({tokens[1] for tokens in reference_tokens})),
                'smer': ('levo', 'desno', 'gore', 'dole', 'napred', 'nazad'),
                'slovo_2': tuple(sorted({tokens[3] for tokens in reference_tokens})),
                'dan': ('ponedeljak', 'utorak', 'sreda', 'četvrtak', 'petak', 'subota', 'nedelja'),
                'broj': ('nula', 'jedan', 'dva', 'tri', 'četiri', 'pet', 'šest', 'sedam', 'osam', 'devet'),
            }
            final_predictions = predictions_by_decoder['beam_w50_lm_a1.0_b0.5']
            final_confusions = slot_confusion_analysis(
                final_predictions,
                references,
                slot_names=SLOT_NAMES,
                vocabularies=slot_vocabularies,
                other_label='ostalo',
                deletion_label='brisanje',
            )
            assert final_confusions['eligible_samples'] == 540

            selected_slots = ('komanda', 'smer', 'dan', 'broj')
            figure, axes = plt.subplots(2, 2, figsize=(17, 13), layout='constrained')
            for axis, slot_name in zip(axes.flat, selected_slots):
                payload = final_confusions['slots'][slot_name]
                full_counts = np.asarray(payload['counts'])
                visible_columns = np.flatnonzero(full_counts.sum(axis=0) > 0)
                matrix = np.asarray(payload['row_normalized'])[:, visible_columns]
                prediction_labels = [payload['prediction_labels'][index] for index in visible_columns]
                image = axis.imshow(matrix, cmap='Blues', vmin=0, vmax=1, aspect='auto')
                axis.set_title(slot_name.capitalize(), loc='left', weight='bold')
                axis.set_xticks(
                    range(len(prediction_labels)), prediction_labels,
                    rotation=35, ha='right', rotation_mode='anchor',
                )
                axis.set_yticks(range(len(payload['reference_labels'])), payload['reference_labels'])
                axis.set_xlabel('Predikcija')
                axis.set_ylabel('Referenca')
                for row in range(matrix.shape[0]):
                    for column in range(matrix.shape[1]):
                        value = matrix[row, column]
                        if value >= 0.01:
                            axis.text(column, row, f'{value:.0%}', ha='center', va='center',
                                      fontsize=8, color='white' if value > 0.55 else COLORS['ink'])
            colorbar = figure.colorbar(
                image, ax=axes.ravel().tolist(), shrink=0.72, pad=0.025,
                label='Udeo unutar referentnog reda',
            )
            colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            figure.suptitle(
                'Matrice konfuzije po semantičkim pozicijama — beam + 5-gram LM',
                x=0.01, ha='left', weight='bold',
            )
            finish_figure(figure, '06_slot_confusion_matrices.png')
            """),
            md("### 10. Najčešće zamene slova"),
            code("""
            figure, axes = plt.subplots(1, 2, figsize=(15, 6))
            for axis, slot_name, title in zip(
                axes, ('slovo_1', 'slovo_2'), ('Prvo slovo', 'Drugo slovo'),
            ):
                substitutions = final_confusions['slots'][slot_name]['substitution_pairs'][:10]
                labels = [f"{item['reference']} → {item['prediction']}" for item in substitutions][::-1]
                values = [item['count'] for item in substitutions][::-1]
                bars = axis.barh(labels, values, color=COLORS['orange'], edgecolor=COLORS['ink'])
                axis.set_title(title, loc='left', weight='bold')
                axis.set_xlabel('Broj poravnatih zamena')
                axis.grid(axis='x', color=COLORS['grid'], alpha=0.7)
                axis.spines[['top', 'right']].set_visible(False)
                axis.bar_label(bars, padding=3)
            figure.suptitle('Najčešće zamene slova — finalni beam+LM decoder',
                            x=0.01, ha='left', weight='bold')
            figure.tight_layout(rect=(0, 0, 1, 0.94))
            finish_figure(figure, '07_letter_substitutions.png')
            """),
            md("### 11. Kvalitativna poboljšanja i regresije"),
            code("""
            greedy_predictions = predictions_by_decoder['greedy']
            lm_predictions = predictions_by_decoder['beam_w50_lm_a1.0_b0.5']
            qualitative_rows = []
            for index, (reference, greedy, lm) in enumerate(
                zip(references, greedy_predictions, lm_predictions)
            ):
                greedy_errors = editdistance.eval(greedy.split(), reference.split())
                lm_errors = editdistance.eval(lm.split(), reference.split())
                qualitative_rows.append({
                    'indeks': index,
                    'promena_broja_gresaka': greedy_errors - lm_errors,
                    'referenca': reference,
                    'greedy': greedy,
                    'beam_5gram': lm,
                })
            improvements = sorted(
                (row for row in qualitative_rows if row['promena_broja_gresaka'] > 0),
                key=lambda row: (-row['promena_broja_gresaka'], row['indeks']),
            )[:3]
            regressions = sorted(
                (row for row in qualitative_rows if row['promena_broja_gresaka'] < 0),
                key=lambda row: (row['promena_broja_gresaka'], row['indeks']),
            )[:3]
            examples = pd.DataFrame(improvements + regressions)
            display(examples.style.hide(axis='index'))
            print('Pozitivna promena znači manje word-edit grešaka sa beam+LM dekoderom.')
            """),
            md("## Provere\n\n### 12. Potvrdi glavne tvrdnje i sve izlazne figure"),
            code("""
            final_metrics = phase7['selected_test_runs']['beam_w50_lm_a1.0_b0.5']['metrics']
            final_bootstrap = phase7['comparisons']['beam_w50_lm_a1.0_b0.5']['paired_bootstrap_vs_greedy']
            assert math.isclose(final_metrics['wer'], 0.41203703703703703, abs_tol=1e-12)
            assert math.isclose(final_metrics['cer'], 0.14696873005743458, abs_tol=1e-12)
            assert final_bootstrap['wer_delta_ci_high'] < 0
            assert final_bootstrap['cer_delta_ci_high'] < 0

            expected_figures = (
                '01_dataset_split.png',
                '02_gpu_demo_mouth_frames.png',
                '03_robustness.png',
                '04_decoder_comparison.png',
                '05_slot_accuracy.png',
                '06_slot_confusion_matrices.png',
                '07_letter_substitutions.png',
            )
            missing_figures = [name for name in expected_figures if not (REPORT_DIR / name).exists()]
            assert not missing_figures, missing_figures
            print('PASS: checkpoint, GPU demo, metrike, intervali i figure su potvrđeni.')
            """),
            md("""
            ## Zaključci

            - Length-aware LipNet baseline ostvaruje test WER 45,25% i CER 18,24%.
            - Ulazne perturbacije menjaju WER za najviše 0,71 procentni poen.
            - Validation-izabrani beam+5-gram decoder spušta WER na 41,20% i CER
              na 14,70% bez promene ili retreninga vizuelnog modela.
            - Paired bootstrap intervali za WER i CER ne obuhvataju nulu.
            - Semantički slotovi su znatno pouzdaniji od dve izolovane pozicije slova.

            Eksperimentalni deo projekta je završen. Potvrđene tabele i figure iz
            ovog notebooka ugrađene su u finalni izveštaj, koji je dostupan u PDF i
            HTML formatu u folderu `report/`.
            """),
        ],
        gpu=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        action="append",
        type=int,
        choices=range(9),
        help="Generiši samo izabranu fazu; opcija može da se ponovi.",
    )
    return parser.parse_args()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    notebooks = {
        0: ("00_faza_0_upstream_restart.ipynb", phase0),
        1: ("01_faza_1_grid_parity.ipynb", phase1),
        2: ("02_faza_2_ai_speak_preprocessing.ipynb", phase2),
        3: ("03_faza_3_serbian_dataset.ipynb", phase3),
        4: ("04_faza_4_transfer_ctc_smoke.ipynb", phase4),
        5: ("05_faza_5_baseline_finetuning.ipynb", phase5),
        6: ("06_faza_6_robustness_experiments.ipynb", phase6),
        7: ("07_faza_7_decoder_search.ipynb", phase7),
        8: ("08_faza_8_konsolidovani_notebook.ipynb", phase8),
    }
    requested = parse_args().phase
    phases = sorted(set(requested)) if requested else list(notebooks)
    for phase in phases:
        name, builder = notebooks[phase]
        value = builder()
        nbf.validate(value)
        nbf.write(value, OUTPUT / name)
        print(OUTPUT / name)


if __name__ == "__main__":
    main()
