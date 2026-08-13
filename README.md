# Vizuelno prepoznavanje govora pomoću LipNet modela

Studentski projekat iz predmeta Mašinsko učenje 2. Cilj praktičnog dela je pokretanje postojeće LipNet implementacije, analiza video pipeline-a, manji eksperimenti sa rezolucijom i augmentacijama i fino podešavanje modela na srpskom delu AI-SPEAK korpusa.

Trenutni status: Faza 1 je završena; implementirana je priprema podataka za Fazu 2.

- [Analiza korpusa, tehničke odluke i roadmap](docs/analiza-i-roadmap.md)
- [Faza 1: uputstvo za Colab/GPU](docs/faza-1-colab.md)
- [Faza 1: Colab notebook](playground/lipnet_faza1_colab.ipynb)
- [Faza 2: manifest, split, vokabular i ROI](docs/faza-2.md)
- [Faza 2: Colab GPU obrada ROI-ja](docs/faza-2-colab-gpu.md)
- [Faza 2: samostalni GPU Colab notebook](playground/lipnet_faza2_gpu_colab.ipynb)
- [Tekst zadatka](36%20Vizuelno%20prepoznavanje%20govora%20na.txt)

Profesor je obezbedio `processed.zip` kao ulazni korpus za projekat. Arhiva, raspakovani `processed/` podaci i težine modela ostaju lokalno i namerno su izuzeti iz Git-a. Realizacija počinje od LipNet inference-a i narednih eksperimentalnih koraka; priprema ili preuzimanje korpusa nisu deo projekta.
