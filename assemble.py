"""

Assemble a corpus of data sentences with mark-up from different corpus.

Three target corpus are created:
- one positive with extended context, including a whole paragraph where at least one annotation is present
- one positive with sentence context
- one negative context corpus, to be used for sampling techniques

We try to further characterize the context with three labels:
- used: is the dataset used in the research work described in the paper 
- created: is the dataset created during the research work described in the paper 
- shared: is the dataset shared and available publicly

Used corpus:

- ner dataset recognition (Heddes et al. 2021) https://doi.org/10.3390/data6080084
https://github.com/xjaeh/ner_dataset_recognition, only NLP/IR/ML domains
- oddpub dataset, more for characterizing mention with respect to "sharing"
- coleridge: the number of dataset is unfortunately very limited and the notion of "dataset" is very loose
- dataseer corpus: high quality annotations including implicit, not named datasets
- transparency indicator dataset: document-level information about dataset but without annotations, and
also restricted to sharing a priori

"""

import os
import json
import argparse
import pandas as pd
from ast import literal_eval
from blingfire import text_to_sentences
import xml
from lxml import etree
import re

def process_ner_dataset_recognition(output):
    """
    This dataset covers named datasets which are reuses, in the IR, ML and NLP domains. 
    It does not cover dataset created for the purpose of a paper (the ones which typically need to be shared too
    for reproducibility, peer reviews,...), so it takes somehow the opposite approach than Dataseer. 
    """
    dataset_path_csv = os.path.join("ner_dataset_recognition", "datasets", "Dataset_sentences.csv")
    df = pd.read_csv(dataset_path_csv) 

    df['labels'] = df['labels'].apply(literal_eval)
    df['datasets'] = df['datasets'].apply(literal_eval)

    #  {‘a’: np.float64, ‘b’: np.int32, ‘c’: ‘Int64’}

    document = {}
    document["lang"] = "en"
    document["level"] = "sentence"
    document["body_text"] = []

    total_annotations = 0
    total_sentences = 0

    # ,id,text,labels,datasets,conference
    # 41,42,"With the close exception of the duration problem in the CAP data sets, PBCMC and BCMC outperformed BMC, sometimes dramatically so.","[[56, 69, 'Dataset']]",['CAP data sets'],NIPS
    for index, row in df.iterrows(): 
        sentence = row['text'].strip()
        labels = row['labels']
        datasets = row['datasets']

        sentence_json = {}
        sentence_json["text"] = sentence
        total_sentences += 1

        annotations = []
        for idx, label in enumerate(labels):
            if len(datasets) > 0:
                dataset = datasets[idx]
            else: 
                dataset = None

            annotation = {}
            annotation["start"] = label[0]
            annotation["end"] = label[1]
            annotation["text"] = sentence[label[0]:label[1]]
            if dataset != None and len(dataset) > 0:
                annotation["type"] = "dataset_name"
                # check with dataset name
                if annotation["text"] != dataset:
                    print(str(row['id']), "- offset issue", ":", annotation["text"], "|", dataset)
            else:
                continue   

            annotations.append(annotation)
            total_annotations += 1

        sentence_json["annotation_spans"] = annotations
        document["body_text"].append(sentence_json)

    output_path = os.path.join(output, "ner_dataset_recognition_sentence.json")
    with open(output_path,'w') as out:
        out.write(json.dumps(document, indent=4))

    print("\ntotal sentences ner_dataset_recognition:", str(total_sentences))
    print("total annotations ner_dataset_recognition:", str(total_annotations))


def process_dataseer(output):
    total_annotations = 0

    '''
    document_paragraph = {}
    document_paragraph["lang"] = "en"
    document_paragraph["level"] = "paragraph"
    document_paragraph["documents"] = []
    '''

    document_sentence = {}
    document_sentence["lang"] = "en"
    document_sentence["level"] = "sentence"
    document_sentence["documents"] = []

    # read csv "dataset" file
    dataset_path_csv = os.path.join("dataseer", "corpus", "csv", "stage1", "All2000_Existing-vs-New_updated_data_types_Feb5_2020.csv")
    df = pd.read_csv(dataset_path_csv, keep_default_na=False) 

    # map every dataset info belong to a document
    document_dataset_map = {}

    # Journal,ArticleNb,DOI,dataset_number,Existing/New,Full_MeSH_data_type,Section,Subsection title,Page number,Column number,Data paragraph,Data_Keyword,Data_action_word,Specialist_equipment,Notes
    for index, row in df.iterrows(): 
        doi = row["DOI"]
        doi = doi.replace("https://doi.org/", "").strip()

        dataset_obj = {}

        dataset_label = row["Data_Keyword"].strip()
        dataset_obj["dataset_label"] = dataset_label

        if len(row["Specialist_equipment"].strip()) > 0:
            acquisition_device_label = row["Specialist_equipment"].strip()
            dataset_obj["acquisition_device_label"] = acquisition_device_label

        dataset_type = row["Full_MeSH_data_type"].strip()
        dataset_obj["dataset_type"] = dataset_type

        '''
        print("dataset_label:", dataset_label)
        print("acquisition_device_label:", acquisition_device_label)
        print("dataset_type:", dataset_type)
        '''

        if not doi in document_dataset_map:
            document_dataset_map[doi] = []

        document_dataset_map[doi].append(dataset_obj)

    # match the datasets into the documents
    for doi in document_dataset_map:

        local_datasets = document_dataset_map[doi]

        if len(local_datasets) == 0:
            continue

        local_document = {}
        local_document["id"] = doi
        local_document["body_text"] = []

        # full text are in TEI XML 
        path_tei_file = os.path.join("dataseer", "corpus", "xml", "stage1", doi.replace("/", "%2F")+".tei.xml")
        if not os.path.exists(path_tei_file):
            #print("cannot find file: ", doi.replace("/", "%2F")+".tei.xml")
            continue

        root = etree.parse(path_tei_file)

        # get all data sentences with xpath
        data_sentences = root.xpath('//tei:s[@id]/text()', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
        for data_sentence in data_sentences:

            data_sentence = data_sentence.replace("[pagebreak]"," ")
            data_sentence = data_sentence.replace("[page break]"," ")
            data_sentence = data_sentence.replace("[columnbreak]"," ")
            data_sentence = data_sentence.replace("[column break]"," ")
            data_sentence = data_sentence.replace("\n"," ")
            data_sentence = re.sub("( )+", " ", data_sentence)
            data_sentence = data_sentence.strip()

            annotations = []
            spans = []

            for local_dataset in local_datasets:
                dataset_label = local_dataset["dataset_label"]
                ind = data_sentence.find(dataset_label)
                if ind != -1:
                    start = ind
                    end = ind + len(dataset_label)

                    annotation = {}
                    annotation["start"] = ind
                    annotation["end"] = ind + len(dataset_label)
                    annotation["text"] = data_sentence[annotation["start"]:annotation["end"]]
                    annotation["type"] = "dataset"
                    annotation["datatype"] = local_dataset["dataset_type"]

                    # check with dataset name
                    if annotation["text"] != dataset_label:
                        print(str(doi), "- offset issue", ":", annotation["text"], "|", dataset_label)
                
                    if overlap(annotation["start"], annotation["end"], spans) == -1:
                        annotations.append(annotation)
                        total_annotations += 1
                        spans.append([annotation["start"], annotation["end"]])

                    if "acquisition_device_label" in local_dataset:
                        acquisition_device_label = local_dataset["acquisition_device_label"]
                        ind2 = data_sentence.find(acquisition_device_label)
                        if ind2 != -1:                        
                            start2 = ind2
                            end2 = ind2 + len(acquisition_device_label)

                            annotation2 = {}
                            annotation2["start"] = ind2
                            annotation2["end"] = ind2 + len(acquisition_device_label)
                            annotation2["text"] = data_sentence[annotation2["start"]:annotation2["end"]]
                            annotation2["type"] = "data_device"
                            annotation2["datatype"] = local_dataset["dataset_type"]

                            if overlap(annotation2["start"], annotation2["end"], spans) == -1:
                                annotations.append(annotation2)
                                total_annotations += 1
                                spans.append([annotation2["start"], annotation2["end"]])
            
            if len(annotations)>0:
                new_section_part = {}
                #new_section_part["section_title"] = section_title
                new_section_part["text"] = data_sentence
                new_section_part["annotation_spans"] = annotations

                local_document["body_text"].append(new_section_part)

        if len(local_document["body_text"])>0:
            document_sentence["documents"].append(local_document)

    output_path = os.path.join(output, "dataseer_sentences.json")
    with open(output_path,'w') as out:
        out.write(json.dumps(document_sentence, indent=4))

    print("\ntotal annotations dataseer:", str(total_annotations))


def process_coleridge(output):
    """
    Coleridge corpus appears very incomplete and cover only a few distinct named datasets (around 65). 
    It does not identify others present datasets.

    We have confusing dataset names and sub-names both present, e.g. National Education Longitudinal Study vs Education Longitudinal Study:

    d0fa7568-7d8e-4db9-870f-f9c6f668c17b,...,National Education Longitudinal Study,National Education Longitudinal Study,national education longitudinal study
    d0fa7568-7d8e-4db9-870f-f9c6f668c17b,...,Education Longitudinal Study,Education Longitudinal Study,education longitudinal study

    (as it does not make much sense, maybe I overlooked a file?)

    Some annotated datasets are actually not dataset, but data repository/project name (managing various datasets),
    e.g. "Alzheimer's Disease Neuroimaging Initiative (ADNI)" which is not a dataset but a research initiative managing various 
    studies and resources (including more than 200 different datasets, but also tools, etc.)
    which adds to the confusion.

    Overall it is either better to skip this corpus for quality reason or to limit it to very short local sentences/segments. 
    """
    total_annotations = 0

    document_sentence = {}
    document_sentence["lang"] = "en"
    document_sentence["level"] = "sentence"
    document_sentence["documents"] = []

    '''
    document_paragraph = {}
    document_paragraph["lang"] = "en"
    document_paragraph["level"] = "paragraph"
    document_paragraph["documents"] = []
    '''

    # read "dataset" train file
    dataset_path_csv = os.path.join("coleridge", "train.csv")
    df = pd.read_csv(dataset_path_csv, keep_default_na=False) 

    # Id,pub_title,dataset_title,dataset_label,cleaned_label

    # Id is actually an identifier of the publication and it is repeated in several raw, one by dataset "form" in this document
    # it seems that we have to match the dataset surface form to the text content
    # no offset, no inline annotation, etc. a sort of archaic/naive format
    
    document_dataset_map = {}

    # we build first a map to get all the datasets of each document
    for index, row in df.iterrows(): 
        doc_id = row["Id"]
        dataset_label = row["dataset_label"]
        dataset_normalized_name = row["dataset_title"]
        dataset_obj = {}
        dataset_obj["dataset_label"] = dataset_label
        dataset_obj["dataset_normalized_name"] = dataset_normalized_name

        if not doc_id in document_dataset_map:
            document_dataset_map[doc_id] = []

        document_dataset_map[doc_id].append(dataset_obj)

    # match the datasets into the documents
    for doc_id in document_dataset_map:

        # this never happens
        if not doc_id in document_dataset_map:
            continue

        local_datasets = document_dataset_map[doc_id]

        if len(local_datasets) == 0:
            continue

        local_document = {}
        local_document["id"] = doc_id
        local_document["body_text"] = []

        # read json document file
        path_json_file = os.path.join("coleridge", "train", doc_id+".json")
        if not os.path.exists(path_json_file):
            path_json_file = os.path.join("coleridge", "test", doc_id+".json")

        if not os.path.exists(path_json_file):
            print("cannot find file: ", doc_id+".json")
            continue

        with open(path_json_file) as json_file:
            json_data = json.load(json_file)
  
            for section_part in json_data:
                section_title = section_part["section_title"]
                section_text = section_part["text"]

                sentence_texts = split_text(section_text)

                for sentence_text in sentence_texts:

                    # list of existing covered spans
                    # used to check if the dataset name is not a sub-string of an already existing
                    # dataset name, or on the contrary extends an existing one
                    spans = []
                    annotations = []

                    for local_dataset in local_datasets:
                        local_dataset_label = local_dataset["dataset_label"]
                        local_dataset_normalized_name = local_dataset["dataset_normalized_name"]
                        ind = sentence_text.find(local_dataset_label)
                        while ind != -1:
                            start = ind
                            end = ind + len(local_dataset_label)

                            annotation = {}
                            annotation["start"] = ind
                            annotation["end"] = ind + len(local_dataset_label)
                            annotation["text"] = sentence_text[annotation["start"]:annotation["end"]]
                            annotation["normalized_name"] = local_dataset_normalized_name

                            # check with dataset name
                            if annotation["text"] != local_dataset_label:
                                print(str(doc_id), "- offset issue", ":", annotation["text"], "|", local_dataset_label)

                            # this will return the index of the first overlapping span or -1
                            span_index = overlap(start, end, spans)
                            if span_index == -1:
                                annotations.append(annotation)
                                spans.append([start, end])
                                total_annotations += 1
                            else:
                                # keep the longest match
                                overlapping_annotation = annotations[span_index]
                                if end-start > overlapping_annotation["end"]-overlapping_annotation["start"]:
                                    annotations[span_index] = annotation

                            ind = sentence_text.find(local_dataset_label, ind+1)

                    if len(annotations)>0:
                        new_section_part = {}
                        new_section_part["section_title"] = section_title
                        new_section_part["text"] = sentence_text
                        new_section_part["annotation_spans"] = annotations

                        local_document["body_text"].append(new_section_part)

            if len(local_document["body_text"])>0:
                document_sentence["documents"].append(local_document)

    output_path = os.path.join(output, "coleridge_sentences.json")
    with open(output_path,'w') as out:
        out.write(json.dumps(document_sentence, indent=4))

    print("\ntotal annotations coleridge:", str(total_annotations))


def process_oddpub(output, fulltext_tei_path="oddpub-dataset/tei"):
    """
    Align csv files with mention information with actual article content. 

    It is supposed that prior to this process, we have downloaded available OA publications
    and process them with Grobid to get usable full text derived from PDF.
    See https://github.com/kermitt2/biblio-glutton-harvester for harvesting PDF
    and https://github.com/kermitt2/grobid
    """

    total_annotations = 0
    map_publications = {}

    document_data_sentence = {}
    document_data_sentence["lang"] = "en"
    document_data_sentence["level"] = "sentence"
    document_data_sentence["documents"] = []

    document_software_sentence = {}
    document_software_sentence["lang"] = "en"
    document_software_sentence["level"] = "sentence"
    document_software_sentence["documents"] = []

    # build map
    with open(os.path.join(fulltext_tei_path, "map.jsonl"), "rt") as mapfile:
        for line in mapfile:
            entry_json = json.loads(line)
            map_publications[entry_json["doi"]] = entry_json["id"]

    dataset_path_csv = os.path.join("oddpub-dataset", "Data1_training_sample_1_results.csv")
    document_data_sentence, document_software_sentence = process_oddpub_file(dataset_path_csv, fulltext_tei_path, map_publications, document_data_sentence, document_software_sentence)

    dataset_path_csv = os.path.join("oddpub-dataset", "Data2_training_sample_2_results.csv")
    document_data_sentence, document_software_sentence = process_oddpub_file(dataset_path_csv, fulltext_tei_path, map_publications, document_data_sentence, document_software_sentence)

    dataset_path_csv = os.path.join("oddpub-dataset", "Data3_validation_sample_results.csv")
    document_data_sentence, document_software_sentence = process_oddpub_file(dataset_path_csv, fulltext_tei_path, map_publications, document_data_sentence, document_software_sentence)

    output_path = os.path.join(output, "oddpub_data_sentences.json")
    with open(output_path,'w') as out:
        out.write(json.dumps(document_data_sentence, indent=4))

    output_path = os.path.join(output, "oddpub_software_sentences.json")
    with open(output_path,'w') as out:
        out.write(json.dumps(document_software_sentence, indent=4))

    print("\ntotal annotations oddpub:", str(total_annotations))


def process_oddpub_file(dataset_path_csv, fulltext_tei_path, map_publications, document_data_sentence, document_software_sentence):
    # read "dataset" file
    
    df = pd.read_csv(dataset_path_csv, keep_default_na=False) 
    
    # depending on the file:
    #doi,oddpub_open_data,oddpub_open_code,open_data_statements,open_code_statements
    # or
    #doi,manual_open_data,manual_open_code,oddpub_open_data,oddpub_open_code,open_data_statements,open_code_statements

    nb_file_found = 0
    for index, row in df.iterrows(): 
        doi = row["doi"]
        doi = doi.replace("https://doi.org/", "").strip()

        if doi in map_publications:
            path_tei_file = os.path.join(fulltext_tei_path, map_publications[doi] + ".tei.xml")

            if not os.path.exists(path_tei_file):
                #print("cannot find file: ", path_tei_file)
                continue

            nb_file_found += 1

            local_document_data = {}
            local_document_data["id"] = doi
            local_document_data["body_text"] = []

            local_document_software = {}
            local_document_software["id"] = doi
            local_document_software["body_text"] = []

            stored_data_sentences = []
            stored_software_sentences = []

            # weird csv formatting, statements are separated with ; but other punctuation (other from delimiter,) 
            # could appear
            # sometimes statements make no sense like 10.1186/s13058-015-0567-2 (raw article header with title and 
            # authors, etc. but nothing related to data or software)

            data_statements = []
            if row["oddpub_open_data"] == True or ("manual_open_data" in row and row["manual_open_data"] == True):
                pre_data_statements = row["open_data_statements"].split(";")
                pre_data_statements = [s.strip() for s in pre_data_statements]

                for data_statement in pre_data_statements:
                    pieces = split_text(data_statement)
                    for piece in pieces:
                        data_statements.append(piece)

            if len(data_statements) == 0:
                continue

            #print(data_statements)

            software_statements = []
            if row["oddpub_open_code"] == True or ("manual_open_code" in row and row["manual_open_code"] == True):
                pre_software_statements = row["open_code_statements"].split(";")
                pre_software_statements = [s.strip() for s in pre_software_statements]

                for software_statement in pre_software_statements:
                    pieces = split_text(software_statement)
                    for piece in pieces:
                        software_statements.append(piece)

            root = etree.parse(path_tei_file)

            # get all data sentences with xpath
            sentences = root.xpath('//tei:s/text()', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'})
            for sentence in sentences:

                sentence = sentence.replace("\n"," ")
                sentence = re.sub("( )+", " ", sentence)
                sentence = sentence.strip()

                if len(sentence) < 20:
                    continue

                simplified_sentence = simplify_text(sentence)

                #print("simplified_sentence:", simplified_sentence)

                for data_statement in data_statements:
                    if len(data_statement) < 20:
                        continue

                    data_statement = data_statement.replace("\n"," ")
                    data_statement = re.sub("( )+", " ", data_statement)
                    simplified_data_statement = simplify_text(data_statement)

                    #print("simplified_data_statement:", simplified_data_statement)
                    
                    if simplified_sentence.startswith(simplified_data_statement) or simplified_sentence.endswith(simplified_data_statement) or simplified_data_statement.startswith(simplified_sentence) or simplified_data_statement.endswith(simplified_sentence): 

                        #annotations = []
                        #spans = []

                        new_section_part = {}
                        #new_section_part["section_title"] = section_title
                        new_section_part["text"] = sentence
                        #new_section_part["annotation_spans"] = annotations

                        if not sentence in stored_data_sentences:
                            local_document_data["body_text"].append(new_section_part)
                            stored_data_sentences.append(sentence)

                for software_statement in software_statements:
                    if len(software_statement) < 20:
                        continue

                    software_statement = software_statement.replace("\n"," ")
                    software_statement = re.sub("( )+", " ", software_statement)
                    simplified_software_statement = simplify_text(software_statement)

                    #print("simplified_data_statement:", simplified_data_statement)
                    
                    if simplified_sentence.startswith(simplified_data_statement) or simplified_sentence.endswith(simplified_data_statement) or simplified_data_statement.startswith(simplified_sentence) or simplified_data_statement.endswith(simplified_sentence): 

                        #annotations = []
                        #spans = []

                        new_section_part = {}
                        #new_section_part["section_title"] = section_title
                        new_section_part["text"] = sentence
                        #new_section_part["annotation_spans"] = annotations

                        if not sentence in stored_software_sentences:
                            local_document_software["body_text"].append(new_section_part)
                            stored_software_sentences.append(sentence)

            if len(local_document_data["body_text"])>0:
                document_data_sentence["documents"].append(local_document_data)

            if len(local_document_software["body_text"])>0:
                document_software_sentence["documents"].append(local_document_software)

    return document_data_sentence, document_software_sentence

def overlap(start, end, spans):
    """
    test if a span overlap a list of existing spans 
    """
    if len(spans) == 0:
        return -1;

    for indx, span in enumerate(spans):
        if (span[0] <= start and start < span[1]) or (span[0] < end and end <= span[1]) or (start < span[0] and span[1] < end):
            return indx
    return -1

def split_text(text):
    return text_to_sentences(text).split('\n')

def simplify_text(text):
    if text == None:
        return ""
    return re.sub("[^a-zA-Z0-9]", "", text).lower()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converter for datasets into json")
    parser.add_argument("--output", type=str, help="path where all the assemble JSON file will be written")
    parser.add_argument("--corpus-type", default=None, type=str, help="type of input corpus")
    
    args = parser.parse_args()
    output = args.output
    corpus_type = args.corpus_type

    if corpus_type == None:
        process_ner_dataset_recognition(output)
        process_dataseer(output)
        process_coleridge(output)
        process_oddpub(output)
    elif corpus_type == 'ner_dataset' or corpus_type == 'ner_dataset_recognition':
        process_ner_dataset_recognition(output)
    elif corpus_type == 'dataseer':
        process_dataseer(output)
    elif corpus_type == 'coleridge':
        process_coleridge(output)
    elif corpus_type == 'oddpub':
        process_oddpub(output)
