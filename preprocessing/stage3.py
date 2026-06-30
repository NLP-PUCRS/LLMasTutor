import pandas as pd
from sklearn.model_selection import train_test_split
import os
import logging
import datetime
import re

class Stage3:
    def __init__(self, df_filtered: pd.DataFrame):
        os.makedirs('log', exist_ok=True)
        
        logging.basicConfig(
            filename=os.path.join('log', f'preprocessing_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'), 
            level=logging.DEBUG, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            encoding='utf-8'
        )
        self.df_input = df_filtered
        self.OUTPUT_DIR = "processed/usnid"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs("rejecteds", exist_ok=True)

        logging.info(f"Stage 3 inicializado com {len(self.df_input)} amostras vindas do Stage 2.")

    def is_noise(self, text):
        """
        Detecta frases que são apenas agradecimentos ou metalinguagem de rebuttal/nota.
        """
        text = str(text).lower().strip()
        patterns = [
            r"thank(s)? (the )?author", 
            r"appreciate the (response|rebuttal|update)",
            r"valuable comments",

            r"change(d)? (the )?rating",
            r"raised (my )?score",
            r"lower(ed)? (my )?score",
            r"rating (has been|to) changed",
            r"updating rating",
            r"change the rating to",
            r"inclined to (accept|reject)",
            r"upgrade (my )?rating",
            r"changed (the )?rating",
            r"overall rating",
            r"author(s)? response",
            r"my concerns",
            r"my decision"
            
            r"^edit:", 
            r"^update:",
            r"author(s)? response",
            r"participating in the discussion",
            r"read the (rebuttal|response)",
            
            r"-->",
            r"->",

        ]
        for pat in patterns:
            if re.search(pat, text):
                return True
        stop_words = {'this', 'is', 'not', 'i', 'the', 'a', 'to', 'be', 'it', 'for', 'of', 'and', 'in'}
        
        clean_words = re.sub(r'[^\w\s]', '', text).split()
        
        meaningful_words = [w for w in clean_words if w not in stop_words]

        if len(meaningful_words) < 4:
            return True
        return False

    def run_refinement(self):
        """Executa apenas a limpeza por Regex (Pente Fino Superficial)"""
        print("\n--- STAGE 3: Pente Fino (Fuzzy/Regex) ---")
        
        self.df_input['is_noise'] = self.df_input['text'].apply(self.is_noise)
        
        df_final_clean = self.df_input[self.df_input['is_noise'] == False].drop(columns=['is_noise']).copy()
        df_rejected_regex = self.df_input[self.df_input['is_noise'] == True].copy()
        
        print(f"Original Stage 2: {len(self.df_input)}")
        print(f"Removidos por Regex: {len(df_rejected_regex)}")
        print(f"Mantidos para USNID: {len(df_final_clean)}")
        
        df_rejected_regex.to_csv('rejecteds/03_regex_rejected_samples.csv', index=False)
        
        return df_final_clean

    def split_usnid(self, df_full):
        """Gera os arquivos train.tsv, dev.tsv e test.tsv formatados para o USNID"""
        logging.debug("Iniciando geração de arquivos TSV para o USNID...")
        
        df_full['label'] = 0
        df_final = df_full[['text', 'label']].copy()

        train_df, temp_df = train_test_split(df_final, test_size=0.2, random_state=42, shuffle=True)
        
        dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

        logging.debug(f"   Train set: {len(train_df)} (80%)")
        logging.debug(f"   Dev set:   {len(dev_df)} (10%)")
        logging.debug(f"   Test set:  {len(test_df)} (10%)")

        train_df.to_csv(os.path.join(self.OUTPUT_DIR, "train.tsv"), sep='\t', index=False)
        dev_df.to_csv(os.path.join(self.OUTPUT_DIR, "dev.tsv"), sep='\t', index=False)
        test_df.to_csv(os.path.join(self.OUTPUT_DIR, "test.tsv"), sep='\t', index=False)
        
        df_full.loc[test_df.index].to_csv(os.path.join("processed", "test_metadata.csv"), index=False)
        
        logging.info("Arquivos TSV salvos com sucesso em processed/usnid/")

    def run(self):
        """Pipeline principal do Stage 3"""
        df_clean = self.run_refinement()
        self.split_usnid(df_clean)
        return df_clean