# Provera rezultata notebookova 02–08

Datum prvog audita: 24. avgust 2026.

Datum završne revalidacije: 25. avgust 2026.

## Zaključak

**Opšta ocena: rezultati Faza 03–07 su potvrđeni i spremni za finalni izveštaj.**

Prvi audit je otkrio da stari Phase 5 run nije bezbedan za citiranje. Nakon toga
su model, metrike, checkpoint tok i notebookovi ispravljeni, a Faze 04–07
izvršene na NVIDIA L4. Novi rezultati koriste checkpoint SHA-256
`203c2707b5c327c8b164ab573f5550390def3aacf0ff190fc9bd760745e2f9c8` i
`corpus-edit-distance-v1` definiciju metrika. Sanitizovane kopije iz Drive-a su:

- [Phase 3 Dataset audit](results/phase3_dataset_audit.json);
- [Phase 4 transfer/CTC audit](results/phase4_transfer_ctc_audit.json);
- [Phase 5 rezultat](results/phase5_results.json);
- [Phase 6 robustnost](results/phase6_robustness_results.json);
- [Phase 7 decoder rezultat](results/decoder_results_v1.json);
- [Phase 7 test reference i predikcije](results/decoder_predictions_v1.json).

Stari brojevi u nastavku ostaju samo kao istorija razloga za revalidaciju.

## Kontrolni izvori

- kod i izlazi sačuvani u izvršenim notebookovima 02–07;
- checkpoint/rezultat putanje prikazane u tim notebookovima na Google Drive-u;
- lokalni testovi nad modelom, Dataset-om, treningom, dekoderom i evaluacijom;
- pinovani VIPL commit
  `40209e09c49553c00c25c7d41faa3706aea3c625`.

Privatni MP4/JPEG podaci i Drive checkpoint-i nisu dostupni u lokalnom
repozitorijumu, pa audit ne može nezavisno da rekonstruiše sve numeričke
rezultate bez ponovnog Colab pokretanja.

## Notebook 02 — preprocessing

Sačuvani izlaz je prikazivao:

- GPU: Tesla T4;
- 3.957 MP4 i 3.957 ALIGN fajlova;
- 186 vraćenih Drive checkpoint arhiva;
- 0 klipova u `failed_clips.log`;
- završnu arhivu `ai_speak_lip.zip` veličine približno 1,34 GiB.

Ovi brojevi su međusobno mogući, ali prethodni restore je vraćao JPEG fajlove
bez spajanja originalnih `decoded_frames`, `landmark_frames` i
`dropped_frames` zapisa. Za vraćene klipove je zato naknadno sintetički
prijavljivano `dropped_frames=0`, pa izbor „graničnih” QA klipova i tvrdnja o
ispuštenim frejmovima nisu pouzdani. Takođe nije postojao programski dokaz da je
ručni QA zaista potvrđen.

Ispravka sada vraća i spaja pravi JSONL audit iz svakog chunk-a, odbija delimične
foldere kao završene, proverava da uspešni i neuspešni zapisi tačno pokrivaju sve
ulaze i zahteva `MANUAL_QA_PASSED=True` pre arhiviranja.

**Status:** postojeći `ai_speak_lip.zip` je zamrznuti ulaz i neće se ponovo
praviti. U završnim tvrdnjama se zato ne navode nepouzdani dropped-frame brojevi;
navodi se samo da je isti postojeći artefakt korišćen u svim narednim fazama.

## Notebook 03 — Dataset i split

Sačuvani izlaz je prikazivao:

- 29 CTC klasa;
- split 16 train, 3 validation i 3 test govornika;
- 2.877 train, 540 validation i 540 test uzoraka (ukupno 3.957);
- primer batch-a oblika `(2, 3, 132, 64, 128)` sa `vid_len=[130,132]` i
  `txt_len=[27,30]`.

Zbir uzoraka se poklapa sa brojem parova iz Faze 2 i speaker liste su
disjunktne. Međutim, stari alfabet round-trip je koristio `.strip()`, čime je
iz testa nenamerno izbačen razmak. Novi notebook eksplicitno testira razmak i
proverava jednakost pronađenih mouth foldera i očekivanih anotacija.

Završni Phase 3 audit potvrđuje 2.877 train, 540 validation i 540 test uzoraka,
29 CTC klasa i očekivani promenljivi batch.

**Status:** potvrđeno; koristi se postojeći Phase 2 artefakt bez ponovnog
preprocessinga.

## Notebook 04 — transfer i CTC smoke

Sačuvani izlaz je prikazivao:

- 22 preneta tenzora;
- namerno preskočene samo `FC.weight` i `FC.bias`;
- batch dužine 58 i 60 frejmova;
- logits `(2, 60, 29)`;
- konačan CTC loss `5.65675687789917` i uspešan backward.

Transfer audit je strukturno dobar. CTC loss je samo smoke vrednost, ne metrika
modela. Pošto je kraći uzorak tada prolazio kroz padded 3D CNN/BiGRU tok, tačna
loss vrednost nije uporediva sa ispravljenim length-aware modelom.

Završni L4 run potvrđuje 22 preneta tenzora, preskočene samo `FC.weight` i
`FC.bias`, logits `(2, 60, 29)`, CTC loss `5.611798286437988` i uspešan backward.

**Status:** potvrđeno u novom length-aware toku.

## Notebook 05 — trening i završne metrike

### Potvrđeni length-aware run

Novi trening je završen do epohe 45. Najbolja je epoha 43 u korisničkom prikazu
(zero-based `best_epoch=42` u JSON-u), sa validation WER `0.4972222222`, CER
`0.2201864147` i loss `0.7389641047`. Jedina test evaluacija najboljeg
checkpoint-a nad 540 uzoraka dala je:

- loss `0.5652946447`;
- WER `0.4524691358`;
- CER `0.1823867262`;
- sentence exact match `0.0`.

Checkpoint hash, konfiguracija, CTC filter i okruženje nalaze se u
[Phase 5 JSON-u](results/phase5_results.json).

**Status:** potvrđeno i bezbedno za kasniji izveštaj uz navedeni protokol i hash.

### Istorijski, nevalidni run

U starom notebooku su zabeleženi sledeći brojevi:

- strong fine-tuning: najbolja epoha 66, validation WER
  `0.5200617283950617`;
- greedy test: WER `0.47222222222222215`, CER
  `0.16367445495476837`, sentence exact match `0.0`, 540 uzoraka;
- character-LM test: WER `0.4351851851851852`, CER
  `0.1587747287811104`.

Ove vrednosti se **ne smeju citirati kao potvrđeni rezultat**, iz sledećih
razloga:

1. Bidirekcioni GRU je obrađivao padded vremenske korake. Dodatno, vremenske 3D
   konvolucije su širile signal u padding, pa je kraći klip davao drugačije
   validne logite kada je bio u batch-u sa dužim klipom.
2. Notebook nije izvršen odozgo nadole. Execution count pokazuje da je strong
   setup pokrenut pre baseline trening ćelije, a postojeći `latest.pt` je već bio
   na epohi 30.
3. Baseline log je zbog duplih vitičastih zagrada štampao literalni tekst
   `{epoch...}` umesto brojeva.
4. Strong trening nije imao pouzdan resume, RNG/history checkpoint ni odvojenu
   verziju protokola; ponovnim pokretanjem počinjao je iznova i mogao da prepiše
   rezultate.
5. Greedy CER je bio prosečna sentence-normalized vrednost, dok je LM CER bio
   corpus zbir grešaka podeljen zbirom karaktera. Njihova razlika zato nije
   validno poboljšanje.
6. Greedy baseline vrednosti za LM poređenje bile su ručno upisane, uključujući
   zaokružen `CER=0.1637`, umesto ponovo izračunate iz istog cache-a.
7. Ručno dodat strong/LM nastavak nije postojao u generatoru notebooka, pa Git
   nije imao jedan reproduktivan izvor.

Stari character-LM dodatak je uklonjen iz baseline notebooka. Novi Faza 5 tok
koristi length-aware model, jednu corpus-level definiciju metrika, novi Drive
folder `phase5_length_aware_v2` i checkpoint hash pri keširanju test rezultata.

**Status istorijskog run-a:** ove stare metrike ostaju nevalidne i ne citiraju se.

## Notebook 06 — robustnost ulaza

Faza 6 je prvo tačno reprodukovala Phase 5 greedy test rezultat, pa je istim
checkpoint-om, splitom i dekoderom izmerila:

| Uslov | WER | CER | Δ WER prema baseline-u |
|---|---:|---:|---:|
| baseline 128×64 | 0.452469 | 0.182387 | 0 |
| 96×48 → 128×64 | 0.457716 | 0.185897 | +0.005247 |
| 64×32 → 128×64 | 0.458951 | 0.186981 | +0.006481 |
| Gaussian blur, 5×5, σ=1 | 0.459568 | 0.187301 | +0.007099 |
| crop shift dx=4, dy=2 | 0.453395 | 0.183408 | +0.000926 |

Sve uočene razlike su male; pošto za perturbacije nisu računati paired
intervali, ne tumače se kao dokaz statistički značajne razlike. Paired
bootstrap je primenjen zasebno u Fazi 7 za poređenje dekodera.

**Status:** potvrđeno; kompletan payload je u
[Phase 6 JSON-u](results/phase6_robustness_results.json).

## Notebook 07 — CTC decoder i jezički model

Faza 7 koristi isti Phase 5 checkpoint i keširane validation/test logit-e, bez
novog treninga ili preprocessinga. Greedy kontrola je egzaktno reprodukovala
Phase 5 test metrike. Karakterni backoff 5-gram LM fitovan je isključivo nad
2.877 train transkripata.

Validation izbor je zaključao beam width 50 bez LM-a i beam width 50 sa
`alpha=1,0`, `word_bonus=0,5` i top-8 token pruning-om. Probe nad 25 uzoraka dao
je potpuno slaganje top-8 i full-alphabet beam predikcija.

| Test dekoder | WER | CER | Exact match |
|---|---:|---:|---:|
| greedy | 0,452469 | 0,182387 | 0,000000 |
| beam width 50, bez LM-a | 0,448765 | 0,181047 | 0,000000 |
| beam width 50 + 5-gram LM | **0,412037** | **0,146969** | **0,003704** |

Za beam+LM u odnosu na greedy, paired bootstrap sa 2.000 iteracija daje:

- Δ WER `-0,040432`, 95% CI `[-0,048148, -0,033025]`;
- Δ CER `-0,035418`, 95% CI `[-0,038405, -0,032550]`;
- verovatnoću boljeg WER-a i CER-a `1,0`.

Intervali za WER i CER ne obuhvataju nulu, pa rezultat podržava zaključak da
izabrani beam+LM decoder pouzdano poboljšava greedy baseline na ovom test skupu.
Analiza pozicija pokazuje najveću tačnost za dan (91,30%), komandu (88,52%),
broj (81,11%) i smer (79,81%), dok izolovana slova ostaju najteži deo rečenice.

**Status:** potvrđeno; rezultati su u
[`decoder_results_v1.json`](results/decoder_results_v1.json), a svih 540
referenci i predikcija u
[`decoder_predictions_v1.json`](results/decoder_predictions_v1.json).

## Notebook 08 — konsolidacija za odbranu

Notebook [`08_faza_8_konsolidovani_notebook.ipynb`](../playground/08_faza_8_konsolidovani_notebook.ipynb)
je implementiran i lokalno validiran kao završni, read-only analitički korak. On
ne trenira vizuelni model i ne ponavlja preprocessing: učitava artefakte Faza
03–07, proverava njihov zajednički checkpoint, izvodi jednu obaveznu GPU
predikciju i generiše sedam figura za izveštaj. Slot-konfuzije eksplicitno vode
brisanja, a reči van očekivanog rečnika svrstavaju u `ostalo`.

**Status:** kod i struktura notebooka su spremni; Colab GPU izvršavanje i vraćanje
notebooka sa outputima još nisu obavljeni.

## Status pre finalnog izveštaja

Eksperimentalne Faze 0–7 su završene, a Faza 08 je implementirana. Preostaje
jedno završno Colab izvršavanje notebooka 08, provera svih sedam PNG izlaza i
vraćanje izvršenog notebooka u repozitorijum. Posle toga nema dodatnih
eksperimenata; jedini preostali korak je finalni izveštaj.
