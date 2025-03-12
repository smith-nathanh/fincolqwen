# the source of train/validation/test sets
# each document needs a corresponding id
# each document must be split into its constituent pages with a page number
# the pages are then made into images (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=596x842> ???)
# use claude 3.7 sonnet (or qwen 2.5 7b?) to generate a query and answer for each document (see template from the colpali repo)
# when creating train/validation/test sets the document id cannot appear in the validation or test sets if it is in the train set, and vice versa
# could consider query types such as open-ended, boolean, compare-contrast, extractive,
# load to hugging face as a dataset with the following columns:
# document_id, page, image_filename, image, query, answer, source, model, prompt


# one master dataset with all the documents and then split it into train, validation, and test sets downsampling everything


# looks like evaluation is done by seeing whether the model retrieved the correct page(s) https://github.com/illuin-tech/vidore-benchmark not whether
# the formulated answer by the LLM interpreting the page was correct.
# create a subclass of BaseVisionRetriever


# Note:
# Answers are generated alongside queries to (1) ground queries and improve their quality and (2) provide
# resources to foster future work


# Query types:
# Extractive: A clear and specific question that can be answered using only a specific piece of information.
# Open-ended: A question that focuses on broad in scope, qualitative aspects of an information.
# Boolean: A yes/no question that may involve multiple steps of reasoning.
# Compare-contrast: A question that requires comparing and/or contrasting two entities or topics that are closely related to each other.
# Enumerative: A question that asks to list all examples that possess a common specific property, optionally requesting details about the specifics of each example.
# Numerical: A question about a specific piece of information that can be calculated using data from the page. The question should require more than simply reading numbers directly from the page.


# sources:
# federal reserve: economic research papers, monetary policy report, financial stability report, economic well beiing report, supervision and regulation report

# oecd economic outlook, policy papers

# CBO budget options report, budget and economic outlook, long term budget outlook, student loan report, distribution of household income

# BLS; employment situation, producer price index, real earnings, us import/export, consumer expenditures

# arxiv quantitative finance

# three largest banks quarterly and annual reports

# Principles of Finance (Dahlquist and Knight, OpenStax)
# Corporate Finance (Ross, Westerfield, Jaffe, Jordan, McGraw-Hill)
# Economics Principles and Practices (Glencoe)
# Principles of Macroeconomics (Greenlaw and Shapiro, OpenStax)
# An Introduction to Quantitative Finance (Blyth, Cambridge University Press)

# each pdf is limited to a maximum of 100 pages

