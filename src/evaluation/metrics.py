import torch
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu
# from bert_score import score # Commented out to avoid heavy dependency for now

def compute_perplexity(model, eval_dataset):
    """
    Compute perplexity on evaluation dataset.
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in eval_dataset:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].size(0)
            total_tokens += batch['attention_mask'].sum().item()
    
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
    return perplexity.item()

def compute_bleu(predictions, references):
    """
    Compute BLEU score.
    """
    bleu = corpus_bleu(predictions, [references])
    return bleu.score

def compute_rouge(predictions, references):
    """
    Compute ROUGE scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    return {k: sum(v)/len(v) for k, v in scores.items()}
