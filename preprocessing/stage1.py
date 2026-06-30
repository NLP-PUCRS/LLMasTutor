import json
import pandas as pd
import re
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import datetime
import nltk
import spacy

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize
spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf")

class Stage1:
    def __init__(self, ARIES_ROOT, S2ORC_FOLDER, REACT_ROOT, ):
        self.ARIES_ROOT = ARIES_ROOT
        self.S2ORC_FOLDER = S2ORC_FOLDER
        self.REACT_ROOT = REACT_ROOT
        self.mapping_dict = {}
        logging.basicConfig(
            filename=os.path.join('log', f'preprocessing_{datetime.datetime.now()}.log'), 
            level=logging.DEBUG, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )

    def process_content_focus(self, text):
        if "[QUOTE_REMOVED]" in text or not text.strip():
            return text

        doc = nlp(text)
        logging.debug(f"process_content_focus->doc={doc}")
        ent_indices = set()
        for ent in doc.ents:
            logging.debug(f"process_content_focus-> ent={ent}")
            if ent.label_ in ['ORG', 'PRODUCT', 'WORK_OF_ART']:
                logging.debug(f"process_content_focus-> ent.label_ ={ent.label_ }")
                for i in range(ent.start, ent.end):
                    ent_indices.add(i)

        tokens = []
        curr_ent = None

        for i, token in enumerate(doc):
            if i in ent_indices:
                ent = token.ent_iob_ # 'B' para início, 'I' para dentro
                if ent == 'B':
                    tokens.append(f"[{token.ent_type_}]")
                continue
                
            if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                tokens.append(token.text)
        
        return " ".join(tokens)
                
    def mapping(self) -> dict:
        """
        Maps docs ids with review ids
        
        :param self: Description
        :return: Description
        :rtype: dict
        """
        doc_to_pdf_map = {}
        split_path = os.path.join(self.ARIES_ROOT, "paper_edits.jsonl")
        try:
            with open(split_path, 'r') as f:
                for line in f:
                    item = json.loads(line)
                    
                    if 'doc_id' in item and 'source_pdf_id' in item:
                        doc_to_pdf_map[item['doc_id']] = item['source_pdf_id']
                        
            logging.debug(f"[Class: {self.__class__.__name__}] - Total mapped docs: {len(doc_to_pdf_map)}")
            logging.debug(list(doc_to_pdf_map.items())[:5])
            self.mapping_dict = doc_to_pdf_map
            return doc_to_pdf_map

        except FileNotFoundError:
            logging.critical(f"err: {split_path}")
            exit(-1)

        except Exception as e:
            logging.critical(f"err: {e}")
            exit(-1)

    def load_paper_sentences(self, pdf_id):
        """
        FUNÇÃO PARA CARREGAR TEXTO DO PAPER (S2ORC)
        Abre o JSON do paper no S2ORC e retorna um SET com todas as frases do paper.
        Usamos SET para busca rápida (O(1)).
        """
        logging.debug(f"[Class: {self.__class__.__name__}] - loading paper sentences.")
        path = os.path.join(self.S2ORC_FOLDER, f"{pdf_id}.json")

        if not os.path.exists(path):
            return set() 
        try:
            with open(path, 'r') as f:
                paper_data = json.load(f)
                
            full_text_parts = []
            
            if 'abstract' in paper_data['pdf_parse']:
                for section in paper_data['pdf_parse']['abstract']:
                    full_text_parts.append(section['text'])

            if 'body_text' in paper_data['pdf_parse']:
                for section in paper_data['pdf_parse']['body_text']:
                    full_text_parts.append(section['text'])
                    
            full_text = " ".join(full_text_parts).lower()
            full_text = self.normalize_text(full_text)
            sentences = re.split(r'[.!?]', full_text)
            return set(s.strip() for s in sentences if len(s) > 10)
            
        except Exception as e:
            return set()
        
    def quote_removal(self) -> pd.DataFrame:
        """
        PROCESSAMENTO PRINCIPAL (QUOTE REMOVAL)
        
        :param self: Description
        """
        print("2. Processando Comentários e Removendo Citações...")
        logging.debug(f"[Class: {self.__class__.__name__}] - processing comments and removing quotes.")

        comments_path = os.path.join(self.ARIES_ROOT, "review_comments.jsonl")
        processed_reviews = []

        current_pdf_id = None
        current_paper_sentences = set()

        all_comments = []
        with open(comments_path, 'r') as f:
            for line in f:
                all_comments.append(json.loads(line))

        all_comments.sort(key=lambda x: x['doc_id'])

        for entry in tqdm(all_comments):
            doc_id = entry['doc_id']
            raw_comment = entry['comment']
            
            pdf_id = self.mapping_dict.get(doc_id)
            
            if pdf_id and pdf_id != current_pdf_id:
                current_pdf_id = pdf_id
                current_paper_sentences = self.load_paper_sentences(pdf_id)
            
            paper_sentences_list = list(current_paper_sentences)
            comment_sentences = re.split(r'(?<=[.!?])\s+', raw_comment)
            final_sentences = []
            
            for sent in comment_sentences:
                sent = self.normalize_text(sent)
                clean_sent = sent.strip().lower()
                if len(clean_sent) < 15:
                    final_sentences.append(sent)
                    continue

                clean_sent_no_punct = re.sub(r'[^\w\s]', '', clean_sent)
                is_quote = False

                for paper_sent in paper_sentences_list:
                    if clean_sent_no_punct in paper_sent or paper_sent in clean_sent_no_punct:
                        is_quote = True
                        break
                
                if not is_quote:
                    final_sentences.append(sent)
                else:
                    final_sentences.append("[QUOTE_REMOVED]")

            cleaned_text = " ".join(final_sentences)
            
            if len(cleaned_text.replace("[QUOTE_REMOVED]", "")) > 10:
                processed_reviews.append({
                    "text": cleaned_text,
                    "source": "ARIES",
                    "original_id": doc_id
                })
        df_quote_removed = pd.DataFrame(processed_reviews)
        print(f"Total de reviews recuperados e limpos: {len(df_quote_removed)}")
        logging.debug(f"[Class: {self.__class__.__name__}] - total reviews recovered and cleaned:{len(df_quote_removed)}")

        df_quote_removed.to_csv(os.path.join("processed/", "01stage_aries_quote_removal.csv"), index=False)
        return df_quote_removed

    def normalize_text(self, text):

        if not isinstance(text, str): return ""
        text = text.replace('\n', ' ').strip()
        text = re.sub(r'\b(Fig(?:ure)?s?\.?)\s*(\d+[a-z]?|\d+\.\d+)\b', '[FIG]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Tab(?:le)?s?\.?)\s*(\d+[a-z]?|\d+\.\d+)\b', '[TAB]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Sec(?:tion)?s?\.?)\s*\d+(\.\d+)*\b', '[SEC]', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(Eq(?:uation)?s?\.?)\s*\(?\d+[a-z]?(\.\d+)*\)?\b', '[EQ]', text, flags=re.IGNORECASE)
        text = re.sub(r'[-*_]{3,}', '', text)
        if len(re.findall(r'[+\-*/|_^\\{}]', text)) > 3:
            text = "[MATH_EXPRESSION]"
        return text
    
    def is_continuation(self, prev_line, curr_line):
        if not prev_line: return False
        if prev_line[-1] not in ['.', '?', '!']:
            return True
        if curr_line[0].islower():
            return True
        return False
    
    def _unified_candidate_set(self) -> pd.DataFrame:
        logging.debug(f"[Class: {self.__class__.__name__}] - loading aries (return quote_rmemoval)")

        df_aries = self.quote_removal()
        print("Gerando CSV de comparação (Before vs After POS/NER)...")
        comparison_data = []

        def transform_with_log(txt):
            # Normalização original (Regex [FIG], [TAB], etc)
            normalized = self.normalize_text(txt) 
            # Nova transformação (POS + NER)
            final = self.process_content_focus(normalized)
            comparison_data.append({"original": txt, "transformed": final})
            return final

        df_aries['text'] = df_aries['text'].apply(transform_with_log)
        
        # Salva o log comparativo
        df_compare = pd.DataFrame(comparison_data)
        df_compare.to_csv(os.path.join("processed", "debug_transformation_comparison.csv"), index=False)

        #df_aries['text'] = df_aries['text'].apply(self.normalize_text)
        df_aries['source'] = 'ARIES_Edit'
        logging.debug(f"[Class: {self.__class__.__name__}] - {len(df_aries)} linhas do ARIES.")

        logging.debug("2. Carregando ReAct Labeled...")
        react_labeled_path = os.path.join(self.REACT_ROOT, "processed_data.csv")

        try:
            df_react_lbl = pd.read_csv(react_labeled_path)
            df_react_lbl = df_react_lbl[['Text']].rename(columns={'Text': 'text'})
            df_react_lbl['source'] = 'ReAct_Labeled'
            df_react_lbl['text'] = df_react_lbl['text'].apply(self.normalize_text)
            logging.debug(f"[Class: {self.__class__.__name__}] - {len(df_react_lbl)} linhas do ReAct Labeled.")

        except Exception as e:
            logging.error(f"   ERRO ao ler ReAct Labeled: {e}")
            df_react_lbl = pd.DataFrame(columns=['text', 'source'])

        logging.debug("3. Carregando ReAct Unlabeled (txt)...")
        react_unlabeled_path = os.path.join(self.REACT_ROOT, "unlabelled_review_corpus_52k.txt")
        logging.debug("3. Carregando ReAct Unlabeled (Smart Stitching)...")

        try:
            if os.path.exists(react_unlabeled_path):
                react_buffer = []
                current_sentence = ""
                
                with open(react_unlabeled_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    
                    for line in tqdm(lines, desc="Costurando ReAct"):
                        clean_line = line.strip()
                        if not clean_line: continue 
                        clean_line = self.normalize_text(clean_line) 
                        if not clean_line: continue

                        if self.is_continuation(current_sentence, clean_line):
                            current_sentence += " " + clean_line
                        else:
                            if current_sentence and len(current_sentence) > 20:
                                react_buffer.append({
                                    'text': current_sentence,
                                    'source': 'ReAct_Unlabeled',
                                    'original_id': f"unlbl_stitch_{len(react_buffer)}"
                                })
                            current_sentence = clean_line
                    
                    if current_sentence and len(current_sentence) > 20:
                        react_buffer.append({'text': current_sentence, 'source': 'ReAct_Unlabeled', 'original_id': 'last'})
                df_react_unlbl = pd.DataFrame(react_buffer)
                logging.debug(f"   -> {len(df_react_unlbl)} linhas 'costuradas' e recuperadas do ReAct Unlabeled.")

            else:
                logging.warning(f"Aviso: {react_unlabeled_path} não encontrado.")
                df_react_unlbl = pd.DataFrame(columns=['text', 'source'])

        except Exception as e:
            logging.error(f"   ERRO ao ler ReAct Unlabeled: {e}")
            df_react_unlbl = pd.DataFrame(columns=['text', 'source'])

        logging.debug("4. Unificando e Deduplicando...")
        df_full = pd.concat([df_aries, df_react_lbl, df_react_unlbl], ignore_index=True)
        logging.debug(f"   Total Bruto: {len(df_full)}")
        df_full = df_full.drop_duplicates(subset=['text'])
        logging.debug(f"   Total Único: {len(df_full)}")
        df_full.to_csv(os.path.join("processed/", "01stage_unified_candidates_set.csv"), index=False)
        logging.debug("   Backup 'merged_all_reviews.csv' salvo.")
        return df_full