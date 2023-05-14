from transformers import AutoTokenizer, AutoModel
import json
import re

if __name__ == '__main__':
    model_name = "google/flan-t5-xxl"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    samples = json.load(open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\merged_data\inputs.txt"))
    print(len(samples))
    # samples = [
    #     "S", "K", "EOD", "CTX", "Q", "O", "STYLE", "KNOWLEDGE", "END OF CONTEXT", "CONTEXT", "QUESTION", "ONTOLOGY",
    #     "OFFER:(slot5=7:30 am;slot6=$33);INFORM_COUNT:(count=8)", "OFFER:(slot5=[7:30 am];slot6=[$33]), INFORM_COUNT:(count=[8])",
    #     "politely", "polite","EOD", "<EOD>","[EOD]", "END OF CONTEXT", "<END OF CONTEXT>", "[END OF CONTEXT]", " || ", "||"
    # ]
    list_out = []
    tmp = [len(tokenizer.encode(sample, return_tensors='np')[0].tolist()) for sample in samples]
    pattern = r"\<K> (\w+):"
    for i in range(len(tmp)):
        if tmp[i] > 512:
            match = re.search(pattern, samples[i])
            dict_tmp = {"text":samples[i], "length": tmp[i], "domain": match.group(1)}
            list_out.append(dict_tmp)

    with open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\list_out_module_3.txt", 'w') as f:
        json.dump(list_out, f, indent=4)
    print(len(list_out))
    print(max(tmp))
    # model = AutoModel.from_pretrained(model_name)
    # model.config