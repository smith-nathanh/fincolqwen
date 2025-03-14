import argparse
import base64
import json
import os
import time
from io import BytesIO

import anthropic
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from PIL import Image
from tqdm import tqdm

prompt = """
You are an assistant specialized in Multimodal RAG tasks. 

The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus.

The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page.

Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus.

Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.

Generate at most THREE pairs of questions and answers per page in a dictionary with the following format, answer ONLY this dictionary NOTHING ELSE:

{
    "questions": [
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        }
    ]
}
where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers that could be as long as needed.

Note: If there are no questions to ask about the page, return an empty list. Focus on making relevant questions concerning the page.

Here is the page:"""

prompt = """
You are an assistant specialized in Multimodal RAG tasks.

The task is the following: given an image from a pdf page, you will have to generate questions that could be asked by a user to retrieve information from a large documentary corpus - without them knowing precisely what is on a given page. So, don't say things like "what are the first three words on the page?" or "what is the title of the page?" Instead, try to ask questions that could be asked by someone who is trying to retreive some topically relevant information from a large corpus.

The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page.

Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.

Generate at most THREE pairs of questions and answers per page in a dictionary with the following format, answer ONLY this dictionary NOTHING ELSE:

{
    "questions": [
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        },
        {
            "question": "XXXXXX",
            "answer": ["YYYYYY"]
        }
    ]
}
where XXXXXX is the question and ['YYYYYY'] is the corresponding list of answers that could be as long as needed.

IMPORTANT NOTE: If there is very little data on the page (e.g., it is simply a title page, appendix with little information, or blank page), return an empty list in the "questions" key. We will ultimately exclude these pages from the dataset when you return an empty list for these samples.

Here is the page:"""


def encode_image_base64(image_pil):
    """Convert a PIL Image to a base64-encoded string."""
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_qa_for_image(client, image, prompt_text, model_name):
    """Generate Q&A for an image using Claude."""
    try:
        # Convert image to base64 if it's a PIL Image
        if isinstance(image, Image.Image):
            image_b64 = encode_image_base64(image)
        else:
            # Assume it's already a path or bytes that can be converted
            image_b64 = encode_image_base64(Image.open(BytesIO(image)))

        message = client.messages.create(
            model=model_name,
            max_tokens=2000,
            temperature=0.2,
            system="You are a specialized assistant for generating questions and answers from document images.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text + " Respond with valid JSON only.",
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
            # Remove the response_format parameter as it's not supported in your version
        )

        # Parse the response as JSON
        response_content = message.content[0].text
        return json.loads(response_content)

    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return {"questions": []}


def process_split(dataset, client, args, split_name):
    """Process a single split of the dataset while preserving image format."""
    print(f"Processing {split_name} split...")

    # Calculate end index
    end_idx = (
        min(args.start_idx + args.batch_size, len(dataset))
        if args.batch_size > 0
        else len(dataset)
    )

    # Create new columns or prepare to update existing ones
    qa_data_column = [{"questions": []} for _ in range(len(dataset))]
    query_column = (
        dataset["query"]
        if "query" in dataset.column_names
        else ["" for _ in range(len(dataset))]
    )
    answer_column = (
        dataset["answer"]
        if "answer" in dataset.column_names
        else ["" for _ in range(len(dataset))]
    )
    model_column = (
        dataset["model"]
        if "model" in dataset.column_names
        else ["" for _ in range(len(dataset))]
    )

    # Process images
    for idx in tqdm(range(args.start_idx, end_idx)):
        # Get the image from the dataset
        example = dataset[idx]
        image = example[args.image_column]

        # Generate Q&A
        qa_data = generate_qa_for_image(client, image, prompt, args.model)

        # Update new columns with questions and answers
        if qa_data and "questions" in qa_data and len(qa_data["questions"]) > 0:
            first_qa = qa_data["questions"][0]
            # Save all generated questions and answers
            qa_data_column[idx] = qa_data
            # Update first question as query
            query_column[idx] = first_qa["question"]
            # Convert answer array to string if needed
            answer_value = (
                ", ".join(first_qa["answer"])
                if isinstance(first_qa["answer"], list)
                else first_qa["answer"]
            )
            answer_column[idx] = answer_value
        # Always set model field
        model_column[idx] = args.model

        # Rate limiting
        time.sleep(0.5)

    # Add each column individually to preserve the image format
    updated_dataset = dataset

    # Add qa_data column (should be new)
    updated_dataset = updated_dataset.add_column("qa_data", qa_data_column)

    # For existing columns, we need to remove them first and then add updated versions
    if "query" in dataset.column_names:
        updated_dataset = updated_dataset.remove_columns("query")
    updated_dataset = updated_dataset.add_column("query", query_column)

    if "answer" in dataset.column_names:
        updated_dataset = updated_dataset.remove_columns("answer")
    updated_dataset = updated_dataset.add_column("answer", answer_column)

    if "model" in dataset.column_names:
        updated_dataset = updated_dataset.remove_columns("model")
    updated_dataset = updated_dataset.add_column("model", model_column)

    return updated_dataset


def safe_load_from_disk(path):
    """Load dataset from disk with proper file:// protocol handling"""
    if path.startswith("/"):
        path = f"file://{path}"
    return load_from_disk(path)


def main():
    # Add timer start
    start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Generate Q&A from images in a dataset and update the dataset."
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset (local or Huggingface)",
    )
    parser.add_argument(
        "--is_local",
        action="store_true",
        help="Flag to indicate if dataset is stored locally",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train",
        help="Dataset splits to process (comma-separated, or 'all' for all splits)",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="Column containing images"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the updated dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=-1,
        help="Number of examples to process (-1 for all)",
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the dataset to the Hugging Face Hub",
    )
    parser.add_argument(
        "--hf_dataset_name",
        type=str,
        help="Name of the dataset on the Hub (e.g., 'username/dataset_name')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-sonnet-20241022",
        help="Model to use for generating Q&A",
    )
    args = parser.parse_args()

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load the dataset using the safe loader
    if args.is_local:
        print(f"Loading dataset from disk: {args.dataset_path}")
        dataset = safe_load_from_disk(args.dataset_path)
    else:
        print(f"Loading dataset from Huggingface: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)

    # Handle case where dataset is a Dataset (single split) rather than a DatasetDict
    if isinstance(dataset, Dataset):
        # Convert single Dataset to a DatasetDict with a single key (assuming 'train')
        dataset = DatasetDict({"train": dataset})
        print("Converted single Dataset to DatasetDict with 'train' split")

    # Determine which splits to process
    if args.splits.lower() == "all":
        splits_to_process = list(dataset.keys())
    else:
        splits_to_process = [split.strip() for split in args.splits.split(",")]

    print(f"Will process these splits: {splits_to_process}")

    # Process each split
    for split in splits_to_process:
        if split not in dataset:
            print(
                f"Warning: Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}"
            )
            continue

        # Process this split
        processed_split = process_split(dataset[split], client, args, split)

        # Update the dataset with the processed split
        dataset[split] = processed_split

    # Save the updated dataset
    print(f"Saving updated dataset to {args.output_path}")
    dataset.save_to_disk(args.output_path)

    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        if not args.hf_dataset_name:
            print(
                "Warning: --hf_dataset_name is required when using --push_to_hub. Skipping upload."
            )
        else:
            print(f"Pushing dataset to Hugging Face Hub as {args.hf_dataset_name}")
            dataset.push_to_hub(args.hf_dataset_name)
            print(f"Dataset successfully pushed to {args.hf_dataset_name}")

    print(f"Processing complete. Dataset saved to {args.output_path}")

    # Calculate and print total execution time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f} (HH:MM:SS)"
    )


if __name__ == "__main__":
    main()
    # Total execution time: 00:01:19.57 (HH:MM:SS)
# python3 generate_qa.py \
#   --dataset_path /mnt/storage/fincolqwen_data/datasets/testing_dataset \
#   --is_local \
#   --splits train \
#   --output_path /mnt/storage/fincolqwen_data/datasets/testing_dataset \
#   --batch_size 100 \
#   --push_to_hub \
#   --hf_dataset_name smith-nathanh/testing_dataset
