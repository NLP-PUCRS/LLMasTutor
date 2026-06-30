from src.stage1 import Stage1
from src.stage2 import Stage2
from src.stage3 import Stage3
import os
import logging
import pandas as pd

if __name__ == "__main__":
    os.makedirs("log", exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    s1 = Stage1(ARIES_ROOT=os.path.join("datasets", "aries/"), 
                S2ORC_FOLDER=os.path.join("datasets/aries/", "s2orc"), 
                REACT_ROOT=os.path.join("datasets", "ReAct/"))
    s1.mapping()
    df_master = s1._unified_candidate_set()
    logging.debug("stage 1 finished")

    s2 = Stage2(REACT_LABELED=os.path.join("datasets/ReAct/", "processed_data.csv"),
                UNIFIED_CANDIDATE_SET=df_master)
    
    df_filtered = s2.run()
    logging.debug("stage 2 finished")
    
    s3 = Stage3(df_filtered=df_filtered)
    df_final_usnid = s3.run()
    logging.debug("stage 3 finished")

