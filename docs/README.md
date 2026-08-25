# Dokumentacija projekta

Ovaj folder okuplja metodologiju, tehničke odluke i proverene rezultate na
kojima se zasniva finalni rad.

## Sadržaj

| Dokument | Opis |
|---|---|
| [`analiza-i-roadmap.md`](analiza-i-roadmap.md) | metodologija, faze eksperimenta i kriterijumi završetka |
| [`provera-rezultata.md`](provera-rezultata.md) | audit rezultata i završna Colab provera notebooka 08 |
| [`upstream-diff.md`](upstream-diff.md) | poreklo koda i dokumentovana prilagođavanja VIPL LipNet-a |
| [`results/`](results/README.md) | sanitizovani JSON auditi i finalne numeričke metrike |
| [`../report/`](../report/README.md) | finalni PDF/HTML izveštaj i verzionisane figure |

## Kako čitati dokumentaciju

Za brz pregled projekta početi od [glavnog README-a](../README.md), a zatim:

1. otvoriti `analiza-i-roadmap.md` za dizajn kompletnog eksperimenta;
2. koristiti `results/` kao primarni izvor brojčanih rezultata;
3. pogledati `provera-rezultata.md` za trag validacije, uključujući završni
   `PASS` notebooka 08;
4. konsultovati `upstream-diff.md` kada je potrebno razlikovati izvorni VIPL
   kod od prilagođavanja za srpski AI-SPEAK korpus.

Završni narativ i figure nalaze se u folderu [`report`](../report/README.md).

Dokumenti ne sadrže privatne snimke, frejmove usana ni težine modela.
