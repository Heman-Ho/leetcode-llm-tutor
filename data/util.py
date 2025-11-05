import re
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer
from huggingface_hub import login
from transformers import AutoModelForSequenceClassification

tokenizer=AutoTokenizer.from_pretrained("Salesforce/codet5-small")
tokenizer=AutoTokenizer.from_pretrained("Salesforce/codet5-small")


def extract_data(problem_description: str) -> list:
    """
    Extract example Input/Output pairs from a problem description using regex.

    Returns:
        A list of dictionaries of the form: [{"input": "...", "output": "..."}, ...]

    Behavior and notes:
    - Uses a regex that looks for optional "Example <n>:" blocks (case-insensitive),
      followed by an "Input:" label and an "Output:" label. It extracts the text
      between Input: and Output: as the example input and the text after Output:
      up to the next Example: or end-of-string as the example output.
    - Keeps the extracted text largely as-is (stripped of surrounding whitespace) so
      it remains suitable as a Seq2Seq target.
    """

    examples = []
    if not problem_description:
        return examples

    # Regex explanation:
    # (?:Example\s*\d*\s*:)? -> optional 'Example' header like 'Example 1:'
    # \s*Input\s*:\s* -> the 'Input:' label (whitespace tolerant)
    # (.*?) -> non-greedy capture of the input block
    # \s*Output\s*:\s* -> the 'Output:' label
    # (.*?)(?=(?:\n\s*Example\s*\d*\s*:)|\Z) -> non-greedy capture of the output block up to next Example or end
    pattern = re.compile(
        r'(?:Example\s*\d*\s*:)?\s*Input\s*:\s*(.*?)\s*Output\s*:\s*(.*?)(?=(?:\n\s*Example\s*\d*\s*:)|\Z)',
        flags=re.IGNORECASE | re.DOTALL
    )

    for m in pattern.finditer(problem_description):
        raw_input = m.group(1).strip()
        raw_output = m.group(2).strip()

        # Append as a simple dict; keep values as clean text for tokenizer.
        examples.append({"input": raw_input, "output": raw_output})

    return examples

def transform_data(problem_description: str) -> str:
    """
    Build a clear prompt instructing the model to generate example input/output pairs
    from the provided problem description.

    Returns:
        A single string prompt that includes an instruction and the formatted problem description.
    """

    # Instruction and formatting guidance for the model.
    prompt = (
        "Given this problem description, generate valid examples (input/output pairs) that match the problem requirements.\n\n"
    )
    prompt += "Problem Description:\n"
    prompt += problem_description.strip() + "\n\n"
    prompt += (
        "Format the examples like this:\n"
        "Example 1:\n"
        "Input: <input here>\n"
        "Output: <output here>\n\n"
    )
    prompt += "Provide multiple diverse examples when applicable."

    return prompt

def store_data(data: list, excel_path: str = "data/examples.xlsx") -> dict:
    """
    Store processed problem/example data to an Excel file.

    Args:
        data: A list of records. Each record can be:
            - a dict with keys: 'problem_description' (str) and optional 'examples' (list of {"input","output"}),
            - or a raw string (treated as the problem_description).
        excel_path: Path where the Excel file will be written. Parent directories will be created.

    Returns:
        A dict with a simple summary: {'status': 'ok'|'error', 'details': ...}

    Notes:
    - Each example becomes one row with columns: problem_id, problem_description, example_index, input, output.
    """

    if not data:
        return {"status": "ok", "details": "no data provided"}

    rows = []
    for pid, record in enumerate(data, start=1):
        if isinstance(record, str):
            desc = record
            examples = extract_data(desc)
        elif isinstance(record, dict):
            desc = record.get("problem_description", "")
            examples = record.get("examples")
            if examples is None:
                examples = extract_data(desc)
        else:
            # skip unsupported types
            continue

        if not examples:
            rows.append({
                "problem_id": pid,
                "problem_description": desc,
                "example_index": 0,
                "input": "",
                "output": ""
            })
        else:
            for i, ex in enumerate(examples, start=1):
                rows.append({
                    "problem_id": pid,
                    "problem_description": desc,
                    "example_index": i,
                    "input": ex.get("input", ""),
                    "output": ex.get("output", "")
                })

    try:
        import os
        try:
            import pandas as pd
        except Exception as e:
            return {"status": "error", "details": f"pandas is required to write Excel files: {e}"}

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)

        df = pd.DataFrame(rows)
        df.to_excel(excel_path, index=False)
        return {"status": "ok", "details": f"wrote {len(df)} rows to {excel_path}"}
    except Exception as e:
        return {"status": "error", "details": str(e)}


def preprocess_dataset(batch):
    """
    Prepare tokenized model inputs/labels for Seq2Seq training.

    - Model input: the prompt produced by transform_data(problem_description)
    - Model target: all examples extracted from the problem_description joined into
      a single string in the form:

        Example 1:\nInput: ...\nOutput: ...\n\nExample 2: ...

    If no examples are found, a short placeholder is used so labels are not empty.
    """

    input_texts = []
    target_texts = []

    # Iterate over problem descriptions in the batch
    for problem_description in batch["problem_description"]:
        # Build the model input prompt from the full problem description
        input_text = transform_data(problem_description)
        input_texts.append(input_text)

        # Extract structured examples from the description
        examples = extract_data(problem_description)

        # Join examples into one target string suitable for Seq2Seq tokenization
        if examples:
            joined = []
            for i, ex in enumerate(examples, start=1):
                joined.append(f"Example {i}:\nInput: {ex['input']}\nOutput: {ex['output']}")
            target_text = "\n\n".join(joined)
        else:
            # Placeholder when no explicit examples exist in the description
            target_text = "No examples provided."

        target_texts.append(target_text)

    # Tokenize inputs and targets for Seq2Seq training
    model_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    labels = tokenizer(
        target_texts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
