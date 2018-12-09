import sys
import os
import pandas as pd
import ntpath
SUMMARYSTUFF = """
## Contents
{:.no_toc}
*  
{: toc}
"""
filetoread = sys.argv[1]
fdtoread = open(filetoread)
print(filetoread)
fileprefix = ".".join(filetoread.split('.')[:-1])
filetowrite = fileprefix+".newmd"
buffer = ""
title = []
for line in fdtoread:
    if line[0:2]=='# ' and not title:#assume title
        title = line.strip()[2:]
    else:
        buffer = buffer + line
fdtoread.close()

# get file meta data
fname = ntpath.basename(filetoread)
fname=fname.split(".")[0]
meta_df = pd.read_csv("../_layouts/page_meta_data.csv")
meta_data = meta_df.loc[meta_df.filename==fname]

# configure YAML preamble
preamble = "title: {}\nnotebook: {}\nsection: {}\nsubsection: {}\n"\
    .format(title, fileprefix+".ipynb", meta_data.section.values[0], meta_data.subsection.values[0])
preamble = "---\n"+preamble+"---\n"

# write new file
fdtowrite=open(filetowrite, "w")
summarystuff = SUMMARYSTUFF
fdtowrite.write(preamble+summarystuff+buffer)
fdtowrite.close()

try:
    os.rename(filetowrite, filetoread)
except WindowsError:
    os.remove(filetoread)
    os.rename(filetowrite, filetoread)
