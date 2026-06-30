import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
import os
import logging
import datetime

class Stage2:
    def __init__(self, REACT_LABELED, UNIFIED_CANDIDATE_SET):
        self.REACT_LABELED = REACT_LABELED
        self.UNIFIED_CANDIDATE_SET = UNIFIED_CANDIDATE_SET
        self.OUTPUT_FILTERED = "processed/02-finetuning-roberta.csv"
        self.MODEL_NAME = "distilroberta-base"
        logging.basicConfig(
            filename=os.path.join('log', f'preprocessing__{datetime.datetime.now()}.log'), 
            level=logging.DEBUG, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.df_raw = pd.read_csv(self.REACT_LABELED)
        logging.debug("\n--- Distribuição Original ReAct ---")
        logging.debug(self.df_raw.groupby(['Label1', 'Label2']).size())
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.trainer = None
        self.training_args = TrainingArguments(
            output_dir="./output_model",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=50,
            load_best_model_at_end=False,
            #metric_for_best_model="accuracy"
            save_total_limit=1
        )

    def create_smart_label(self, row):
        """Mapeia as labels do ReAct para binário (Actionable=1, Ruído=0)"""
        l1 = str(row['Label1']).lower().strip()
        if l1 == 'actionable':
            return 1
        return 0
    
    def metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": (predictions == labels).mean()}
    
    def tokenize_function(self, samples):
        return self.tokenizer(samples["text"], padding="max_length", truncation=True, max_length=128)

    def prepare_data(self):
        """Prepara o dataset para o formato do HuggingFace"""
        df = self.df_raw.copy()
        df['labels'] = df.apply(self.create_smart_label, axis=1)
        df = df[['Text', 'labels']].rename(columns={'Text': 'text'})
        df = df.dropna(subset=['text'])
        
        logging.debug("\n--- Distribuição Final para Treino ---")
        logging.debug(df['labels'].value_counts())
        train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['labels'])
        train_df, val_df = train_test_split(train_val_df, test_size=0.176, random_state=42, stratify=train_val_df['labels'])  
        logging.debug(f"Distribuição: Treino={len(train_df)}, Val={len(val_df)}, Teste={len(test_df)}")
        
        self.train_dataset = Dataset.from_pandas(train_df).map(self.tokenize_function, batched=True)
        self.val_dataset = Dataset.from_pandas(val_df).map(self.tokenize_function, batched=True)
        self.test_dataset = Dataset.from_pandas(test_df).map(self.tokenize_function, batched=True)
        
        logging.info(f"Splits: Treino={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    def _train(self):
        """Executa o Fine-tuning do DistilRoBERTa"""
        logging.debug(f"\nIniciando Treinamento com {self.MODEL_NAME}...")
        model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME, num_labels=2)
        self.trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.metrics,
        )
        
        self.trainer.train()
        eval_res = self.trainer.evaluate()
        logging.debug(f"\nAcurácia Final Validação: {eval_res['eval_accuracy']:.2%}")
        test_results = self.trainer.predict(self.test_dataset)
        test_acc = test_results.metrics['test_accuracy']
        logging.info(f"--- RESULTADO FINAL PARA O PAPER ---")
        logging.info(f"Acurácia no Test Set (Hold-out): {test_acc:.2%}")
    
    def _report_test_accuracy(self):
        """Este é o número que você vai colocar no seu artigo"""
        logging.info("--- AVALIANDO NO TEST SET (HOLD-OUT) ---")
        results = self.trainer.predict(self.test_dataset)
        acc = results.metrics['test_accuracy']
        print(f"\n[PAPER RESULT] Accuracy on Test Set: {acc:.2%}")
        logging.info(f"Accuracy on Test Set: {acc:.2%}")

    def _inference_n_save(self):
        """Aplica o filtro no dataset mestre (ARIES + ReAct Unlabeled)"""
        logging.debug("\n--- INFERÊNCIA EM MASSA (FILTRAGEM) ---")
        df_full = self.UNIFIED_CANDIDATE_SET.dropna(subset=['text']).copy()
        
        ds_full = Dataset.from_pandas(df_full[['text']]).map(self.tokenize_function, batched=True)
        
        raw_preds = self.trainer.predict(ds_full)
        df_full['is_actionable'] = np.argmax(raw_preds.predictions, axis=-1)

        df_clean = df_full[df_full['is_actionable'] == 1].copy()
        
        logging.debug(f"Total Original: {len(df_full)}")
        logging.debug(f"MANTIDOS (Actionable): {len(df_clean)} ({len(df_clean)/len(df_full):.1%})")

        df_clean.to_csv(self.OUTPUT_FILTERED, index=False)
        logging.debug(f"Dataset filtrado salvo em: {self.OUTPUT_FILTERED}")
        return df_clean

    def split_usnid(self, df_full):
        logging.debug("5. Gerando arquivos TSV finais para o USNID...")
        df_full['label'] = 0
        df_final = df_full[['text', 'label']]

        train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42, shuffle=True)
        dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
        logging.debug(f"   Train set: {len(train_df)} (80%)")
        logging.debug(f"   Dev set:   {len(dev_df)} (10%)")
        logging.debug(f"   Test set:  {len(test_df)} (10%) -> Perfeito para t-SNE")

        os.makedirs("processed/", exist_ok=True)
        train_df.to_csv(os.path.join("processed/", "train.tsv"), sep='\t', index=False)
        dev_df.to_csv(os.path.join("processed/", "dev.tsv"), sep='\t', index=False)
        test_df.to_csv(os.path.join("processed/", "test.tsv"), sep='\t', index=False)
        df_full.loc[test_df.index].to_csv(os.path.join("processed/", "test_metadata.csv"), index=False)
        print(f"\nArquivos prontos em processed/..")
    
    def run(self):
        self.prepare_data()
        self._train()
        self._report_test_accuracy()
        df_filtered = self._inference_n_save()
        return df_filtered