from transformers import BertTokenizer
import torch
import emoji
import nltk


def create_rationale_mask(input_ids: torch.Tensor, dog_whistle: str,
                          tokenizer: BertTokenizer) -> torch.Tensor:
    """Create a rationale mask identifying dog whistle tokens."""
    rationale_mask = torch.zeros_like(input_ids)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    # Tokenize dog whistle
    dog_whistle_encoding = tokenizer(dog_whistle,
                                     add_special_tokens=False,
                                     return_tensors="pt")
    dog_whistle_tokens = tokenizer.convert_ids_to_tokens(
        dog_whistle_encoding["input_ids"][0].tolist())

    # Find where dog whistle tokens appear in the text
    for i, token in enumerate(all_tokens):
        if token.startswith("[") and token.endswith("]"):
            continue

        # Strategy 1: Direct token match or substring match
        if any(token.lower() == dwt.lower() or dwt.lower() in token.lower()
               or token.lower() in dwt.lower() for dwt in dog_whistle_tokens):
            rationale_mask[i] = 1
            continue

        # Strategy 2: Check for embedded dog whistle in concatenated tokens
        window_size = min(len(dog_whistle_tokens) + 2, len(all_tokens) - i)
        window_tokens = all_tokens[i:i + window_size]
        joined_window = ''.join(t.lower().replace('##', '')
                                for t in window_tokens)

        if dog_whistle.lower(
        ) in joined_window or joined_window in dog_whistle.lower():
            rationale_mask[i:i + window_size] = 1
            continue

        # Strategy 3: For acronyms and short phrases, try partial matches
        if len(dog_whistle_tokens) <= 3:
            window_size = len(dog_whistle_tokens)
            window_tokens = all_tokens[i:i + window_size]

            matches = sum(
                1 for w, d in zip(window_tokens, dog_whistle_tokens)
                if (w.lower().startswith(d.lower()) or d.lower().startswith(
                    w.lower()) or w.lower().replace('##', '') in d.lower()
                    or d.lower() in w.lower().replace('##', '')))

            if matches > 0:
                rationale_mask[i:i + window_size] = 1

    return rationale_mask


def extract_relevant_sentence(text: str, dog_whistle: str) -> str:
    """Extract the sentence containing the dog whistle."""
    # Split text into sentences
    if not any(c.isalnum() for c in dog_whistle):
        # For purely special character dog whistles, split text into smaller chunks
        sentences = nltk.sent_tokenize(text)
        if len(sentences) == 1:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) == 1 and len(text) > 512:
                splits = [
                    s.strip() for s in text.replace('\n', '.').split('.')
                    if s.strip()
                ]
                if splits:
                    sentences = splits
    else:
        sentences = nltk.sent_tokenize(text)

    # Find the sentence containing the dog whistle
    target_sentence = None
    dog_whistle_lower = dog_whistle.lower()

    for sentence in sentences:
        # Create text variations for matching
        sentence_variations = [
            sentence,
            sentence.lower(),
            ''.join(sentence.lower().split()),
        ]

        # Special handling for purely special character dog whistles
        if not any(c.isalnum() for c in dog_whistle):
            sentence_variations.append(sentence.replace(' ', ''))
        else:
            sentence_variations.append(''.join(c for c in sentence.lower()
                                               if c.isalnum()))

        dog_whistle_variations = [
            dog_whistle,
            dog_whistle_lower,
            ''.join(dog_whistle_lower.split()),
        ]

        if any(c.isalnum() for c in dog_whistle):
            dog_whistle_variations.append(''.join(c for c in dog_whistle_lower
                                                  if c.isalnum()))

        if any(
                any(dw in sv for sv in sentence_variations)
                for dw in dog_whistle_variations):
            target_sentence = sentence
            break

    # If no direct match, try finding partial matches
    if target_sentence is None and len(sentences) > 1:
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            for part in dog_whistle_lower.split():
                if part in sentence_lower:
                    start_idx = max(0, i - 1)
                    end_idx = min(len(sentences), i + 2)
                    target_sentence = ' '.join(sentences[start_idx:end_idx])
                    break
            if target_sentence:
                break

    return target_sentence if target_sentence else (
        sentences[0] if len(text) > 512 else text)


def generate_rationale_mask(item, tokenizer, max_length):
    """Process a single item from the dataset."""

    # Skip the item with HH dog whistle
    if item["dog_whistle"] == "HH":
        return None
    
    # Convert emojis to text representation
    text = emoji.demojize(item["content"], delimiters=("", ""))
    dog_whistle = emoji.demojize(item["dog_whistle"], delimiters=("", ""))

    # Get the sentence containing the dog whistle
    text = extract_relevant_sentence(text, dog_whistle)

    # Tokenize text
    encoding = tokenizer(text,
                         max_length=max_length,
                         padding="max_length",
                         truncation=True,
                         return_tensors="pt")

    # Create rationale mask
    rationale_mask = create_rationale_mask(encoding["input_ids"][0],
                                           dog_whistle, tokenizer)

    return {
        "content": text,
        "dog_whistle": dog_whistle,
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "rationale_mask": rationale_mask,
    }
