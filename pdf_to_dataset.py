import gc
import os
import uuid
from typing import List, Optional, Tuple

import pdf2image
from datasets import Dataset, Features, Image, Value, concatenate_datasets


def generate_document_id() -> str:
    """Generate a unique document ID."""
    return str(uuid.uuid4())


def convert_pdf_to_images(
    pdf_path: str, output_dir: str, dpi: int = 300
) -> List[Tuple[str, int]]:
    """
    Convert PDF pages to images and save them to output directory.
    Processes one page at a time to reduce memory usage.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the images
        dpi: Resolution for the converted images

    Returns:
        List of tuples containing (image_path, page_number)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract PDF filename without extension to use in image filenames
    pdf_filename = os.path.basename(pdf_path)
    pdf_name = os.path.splitext(pdf_filename)[0]

    saved_images = []

    try:
        # Get the total number of pages
        info = pdf2image.pdfinfo_from_path(pdf_path)
        total_pages = info["Pages"]

        print(f"Processing {pdf_name} with {total_pages} pages")

        # Process each page individually
        for page_num in range(1, total_pages + 1):
            # Convert a single page
            try:
                images = pdf2image.convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    fmt="jpeg",
                    first_page=page_num,
                    last_page=page_num,
                )

                if images and len(images) > 0:
                    image = images[0]
                    image_filename = f"{pdf_name}_page_{page_num}.jpg"
                    image_path = os.path.join(output_dir, image_filename)

                    # Save the image
                    image.save(image_path, "JPEG")
                    saved_images.append((image_path, page_num))

                    # Free memory
                    del image
                    del images
                    gc.collect()

                    # Print progress occasionally
                    if page_num % 10 == 0 or page_num == total_pages:
                        print(
                            f"  Processed page {page_num}/{total_pages} of {pdf_name}"
                        )

            except Exception as e:
                print(f"Error converting page {page_num} of {pdf_filename}: {e}")
                continue

    except Exception as e:
        print(f"Error getting PDF info for {pdf_filename}: {e}")

    gc.collect()
    return saved_images


def process_pdfs_to_dataset(
    pdf_dir: str,
    output_dir: str,
    image_output_dir: Optional[str] = None,
    output_file: str = "pdf_dataset",
    batch_size: int = 3,  # Process a small number of PDFs at a time
    max_pages_per_pdf: Optional[
        int
    ] = None,  # Limit the number of pages processed per PDF
) -> Dataset:
    """
    Process all PDFs in a directory to create a HuggingFace dataset.
    Uses batching to reduce memory usage.

    Args:
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save the dataset
        image_output_dir: Directory to save the extracted images (if None, uses temp dir)
        output_file: Name of the output directory
        batch_size: Number of PDFs to process in each batch
        max_pages_per_pdf: Maximum number of pages to process from any PDF (None for all pages)

    Returns:
        HuggingFace dataset
    """
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)

    if image_output_dir is None:
        image_output_dir = os.path.join(output_dir, "images")

    os.makedirs(image_output_dir, exist_ok=True)

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    print(f"Found {len(pdf_files)} PDF files to process")

    # Create features once
    features = Features(
        {
            "document_id": Value("string"),
            "page": Value("int64"),
            "image_filename": Value("string"),
            "image": Image(),  # Using the Image feature
            "query": Value("string"),
            "answer": Value("string"),
            "source": Value("string"),
            "model": Value("string"),
            "prompt": Value("string"),
        }
    )

    # Process PDFs in batches
    all_datasets = []

    for i in range(0, len(pdf_files), batch_size):
        batch_pdfs = pdf_files[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(pdf_files) + batch_size - 1)//batch_size}: {len(batch_pdfs)} PDFs"
        )

        # Lists to store dataset rows for this batch
        document_ids = []
        pages = []
        image_filenames = []
        image_paths = []
        queries = []
        answers = []
        sources = []
        models = []
        prompts = []

        # Process each PDF file in the batch
        for pdf_file in batch_pdfs:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            doc_id = generate_document_id()

            try:
                # Get PDF info to check page count
                try:
                    info = pdf2image.pdfinfo_from_path(pdf_path)
                    total_pages = info["Pages"]

                    # Apply page limit if specified
                    if max_pages_per_pdf and total_pages > max_pages_per_pdf:
                        print(
                            f"Limiting {pdf_file} to first {max_pages_per_pdf} pages (has {total_pages} total)"
                        )
                        # Modify pdf_path or pass max_pages to convert_pdf_to_images
                        last_page = max_pages_per_pdf
                    else:
                        last_page = total_pages

                    # Convert PDF pages to images (one by one)
                    saved_images = []
                    for page_num in range(1, last_page + 1):
                        try:
                            images = pdf2image.convert_from_path(
                                pdf_path,
                                dpi=300,
                                fmt="jpeg",
                                first_page=page_num,
                                last_page=page_num,
                            )

                            if images and len(images) > 0:
                                image = images[0]
                                image_filename = f"{os.path.splitext(pdf_file)[0]}_page_{page_num}.jpg"
                                image_path = os.path.join(
                                    image_output_dir, image_filename
                                )

                                # Save the image
                                image.save(image_path, "JPEG")
                                saved_images.append((image_path, page_num))

                                # Free memory
                                del image
                                del images
                                gc.collect()
                        except Exception as e:
                            print(
                                f"Error processing page {page_num} of {pdf_file}: {e}"
                            )
                            continue

                    # Process saved images the same as before
                    for image_path, page_num in saved_images:
                        # Add to dataset lists
                        document_ids.append(doc_id)
                        pages.append(page_num)
                        image_filename = os.path.basename(image_path)
                        image_filenames.append(image_filename)
                        image_paths.append(image_path)
                        queries.append("")
                        answers.append("")
                        sources.append(pdf_file)
                        models.append("")
                        prompts.append("")

                except Exception as e:
                    print(f"Error getting info for {pdf_file}: {e}")
                    continue

                # Explicitly run garbage collection after each PDF
                gc.collect()

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

        # Create batch dataset
        try:
            batch_dataset = Dataset.from_dict(
                {
                    "document_id": document_ids,
                    "page": pages,
                    "image_filename": image_filenames,
                    "image": image_paths,  # Paths are automatically loaded as images
                    "query": queries,
                    "answer": answers,
                    "source": sources,
                    "model": models,
                    "prompt": prompts,
                },
                features=features,
            )

            # Save this batch to disk immediately
            batch_path = os.path.join(output_dir, f"batch_{i//batch_size}")
            batch_dataset.save_to_disk(batch_path)

            # Add to our list of datasets to concatenate
            all_datasets.append(batch_path)

            print(f"Batch {i//batch_size + 1} saved with {len(batch_dataset)} entries")

            # Clear memory
            del batch_dataset
            del document_ids, pages, image_filenames, image_paths
            del queries, answers, sources, models, prompts
            gc.collect()

        except Exception as e:
            print(f"Error creating dataset for batch {i//batch_size + 1}: {e}")

    # Load and concatenate all batches
    print("Merging all batches...")
    final_dataset = None

    for i, batch_path in enumerate(all_datasets):
        print(f"Loading batch {i+1}/{len(all_datasets)}")
        batch = Dataset.load_from_disk(batch_path)

        if final_dataset is None:
            final_dataset = batch
        else:
            # Merge batches one by one to avoid loading everything into memory
            final_dataset = concatenate_datasets([final_dataset, batch])

        # Clear batch from memory
        del batch
        gc.collect()

    # Save final dataset
    if final_dataset is not None:
        final_output_path = os.path.join(output_dir, output_file)
        print(
            f"Saving final dataset with {len(final_dataset)} entries to {final_output_path}"
        )
        final_dataset.save_to_disk(final_output_path)

    # Clean up batch directories
    for i in range(len(all_datasets)):
        batch_path = os.path.join(output_dir, f"batch_{i}")
        if os.path.exists(batch_path):
            import shutil

            shutil.rmtree(batch_path)

    return final_dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert PDFs to HuggingFace dataset")
    parser.add_argument(
        "--pdf-dir", type=str, required=True, help="Directory containing PDF files"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save the dataset"
    )
    parser.add_argument(
        "--image-dir", type=str, help="Directory to save extracted images (optional)"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="pdf_dataset",
        help="Name of output dataset directory",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-name",
        type=str,
        help="Dataset name on Hugging Face Hub (username/dataset_name)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Number of PDFs to process in one batch (lower = less memory usage)",
    )
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        help="Maximum number of pages to process from any PDF (None for all pages)",
    )

    args = parser.parse_args()

    # Process PDFs and create dataset
    dataset = process_pdfs_to_dataset(
        pdf_dir=args.pdf_dir,
        output_dir=args.output_dir,
        image_output_dir=args.image_dir,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_pages_per_pdf=args.max_pages_per_pdf,
    )

    if dataset:
        print(f"Created dataset with {len(dataset)} entries")
        print(f"Saved to {os.path.join(args.output_dir, args.output_file)}")

        # Push to Hugging Face Hub if requested
        if args.push_to_hub and args.hub_name:
            dataset.push_to_hub(args.hub_name)
            print(f"Dataset pushed to Hugging Face Hub as {args.hub_name}")


if __name__ == "__main__":
    main()
