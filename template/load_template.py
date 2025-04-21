from transformers import PreTrainedTokenizer

from params import DataArguments

from llamafactory_refs.template import get_template_and_fix_tokenizer


# [TODO] Implement this as part of the training library?
def get_template(tokenizer: "PreTrainedTokenizer", args: "DataArguments"):
	return get_template_and_fix_tokenizer(tokenizer, args)
