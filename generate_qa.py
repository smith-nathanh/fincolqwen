import argparse
import base64
import json
import os
import time
from io import BytesIO

import anthropic
from datasets import load_dataset, load_from_disk
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


def encode_image_base64(image_pil):
    """Convert a PIL Image to a base64-encoded string."""
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def generate_qa_for_image(client, image, prompt_text):
    """Generate Q&A for an image using Claude."""
    try:
        # Convert image to base64 if it's a PIL Image
        if isinstance(image, Image.Image):
            image_b64 = encode_image_base64(image)
        else:
            # Assume it's already a path or bytes that can be converted
            image_b64 = encode_image_base64(Image.open(BytesIO(image)))

        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0.2,
            system="You are a specialized assistant for generating questions and answers from document images.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
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
            response_format={"type": "json_object"},
        )

        # Parse the response as JSON
        response_content = message.content[0].text
        return json.loads(response_content)

    except Exception as e:
        print(f"Error generating Q&A: {e}")
        return {"questions": []}


def process_split(dataset, client, args, split_name):
    """Process a single split of the dataset."""
    print(f"Processing {split_name} split...")

    # Calculate end index
    end_idx = (
        min(args.start_idx + args.batch_size, len(dataset))
        if args.batch_size > 0
        else len(dataset)
    )

    # Process images
    for idx in tqdm(range(args.start_idx, end_idx)):
        # Get the image from the dataset
        example = dataset[idx]
        image = example[args.image_column]

        # Generate Q&A
        qa_data = generate_qa_for_image(client, image, prompt)

        # For debugging
        if idx % 10 == 0:
            print(f"Sample QA data: {qa_data}")

        # Update dataset with questions and answers in-place
        # Extract first question-answer pair (if available)
        if qa_data and "questions" in qa_data and len(qa_data["questions"]) > 0:
            first_qa = qa_data["questions"][0]
            # Save all generated questions and answers
            dataset[idx]["model_qa_data"] = qa_data
            # Save first question as query
            dataset[idx]["query"] = first_qa["question"]
            # Save first answer as model
            dataset[idx]["model"] = first_qa["answer"]
        else:
            dataset[idx]["model_qa_data"] = {"questions": []}
            dataset[idx]["query"] = ""
            dataset[idx]["model"] = []

        # Rate limiting
        time.sleep(0.5)

    return dataset


def main():
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
    args = parser.parse_args()

    # Initialize the Anthropic client
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Load the dataset
    if args.is_local:
        print(f"Loading dataset from disk: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
    else:
        print(f"Loading dataset from Huggingface: {args.dataset_path}")
        dataset = load_dataset(args.dataset_path)

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

    print(f"Processing complete. Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
# python generate_qa.py \
#   --dataset_path /mnt/storage/fincolqwen_data/datasets/pdf_dataset \
#   --is_local \
#   --splits train,validation,test \
#   --output_path /mnt/storage/fincolqwen_data/datasets/pdf_dataset_with_qa \
#   --batch_size 100
