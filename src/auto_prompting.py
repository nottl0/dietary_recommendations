import subprocess
from loguru import logger
import os
import re

"""
Prompt_arguments.txt:
/test/images/13516_jpg.rf.00db1669626040ff9aebc027c1131b2f.jpg 26
/test/images/16299_jpg.rf.698531dc18a266a26474889921db4748.jpg 26
/test/images/11378_jpg.rf.1d815d5c5dcc7627c4ba1b0b45e87496.jpg 26
/test/images/17911_jpg.rf.47876fb503e1c8e1ee01899bd9346de7.jpg 31
/test/images/6754_jpg.rf.f4ebc6635551045e44382640e8fe56a1.jpg 31
/test/images/2580_jpg.rf.f1d8e6cb5c1f3674cf86806372d8f8bf.jpg 31
"""

"""
Settings:
1. pip install loguru
2. set the items below
"""

txt_path = "/workspace/prompt_arguments.txt" # Your txt_path 
script_path = "/workspace/main2.py" # Your script for running
images_path = "/workspace/workspace/data/CAFSD/CAFSD" # Your dataset location 
gpu_number = 9 # You GPU number



command = ["python3", script_path, "-p", "path", "-c", "case"]

with open(txt_path, 'r') as file:
    for line in file:
        columns = line.strip().split()
        if len(columns) == 2:
            col1, col2 = columns
            logger.info(f"Executing -p {col1} -c {col2}")
            command[3] = images_path + col1
            command[5] = col2
            result = subprocess.run(command, capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_number)})

            long_text = result.stdout
            pattern = re.compile(r'\[.*?\]', re.DOTALL)
            match = pattern.search(long_text)

            if match:
                list_str = match.group(0)

                python_list = eval(list_str)

                logger.info("Found python list: " + str(python_list))
            else:
                logger.debug("No list found in the text.")

            logger.debug("Error:", result.stderr)
        else:
            logger.debug(f"Skipping line with unexpected format: {line}")
