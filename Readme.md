# Dataset recognition resources

## Original resources

* Dataseer corpus
* https://github.com/xjaeh/ner_dataset_recognition
* Coleridge corpus
* oddpub dataset https://osf.io/yv5rx/
* transparency-indicators dataset https://osf.io/e58ws/

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

