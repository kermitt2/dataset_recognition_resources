# Dataset recognition resources

## Original resources

Resources and install path of the resources

* Dataseer corpus (`dataseer/`), biomedicine domain, annotations of implicit and explicit datasets, data types and annotation of data acquisition devices
* https://github.com/xjaeh/ner_dataset_recognition (`ner_dataset_recognition/`), IR/ML/NLP domain, only explicitly named and reused datasets
* oddpub dataset https://osf.io/yv5rx/ (`oddpub-dataset/`), biomedicine domain, only article screening (no annotation), only datasets with open access statements, only explicit datasets
* transparency-indicators dataset https://osf.io/e58ws/ (`transparency-indicators-dataset/`), biomedicine domain, only article screening (no annotation)
* Coleridge corpus (`coleridge/`), partial (only a very small subset of "datasets" considered), no explicit annotation, no valid definition of datasets (e.g. research initiative name considered as "dataset")
* SciREX, a dataset of 438 annotated arXiv documents only on ML domain, with identification of datasets (label is "Material"), see https://github.com/allenai/SciREX (reported IAA on 5 documents is 95% average cohen-κ scores), one drawback is the pre-tokenized words which is destructive (because we lose the original delimiters and we can't reconstruct the original text)

## Assemble resources 

Survive in the python dependency marshlands:

```
virtualenv --system-site-packages -p python3.8 env
source env/bin/activate
```

Install dependencies

```sh
pip3 install -r requirements.txt 
```

Assemble resources in the same JSON format: 

```sh
python3 assemble.py --output combined/
```

This will create under `combined/` one JSON file per orginal corpus in the same JSON format using span offsets. 

