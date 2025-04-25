from transformers import PreTrainedTokenizer

from params import DataArguments

from llamafactory_refs.template import get_template_and_fix_tokenizer


def get_template(tokenizer: "PreTrainedTokenizer", args: "DataArguments"):
	return get_template_and_fix_tokenizer(tokenizer, args)
