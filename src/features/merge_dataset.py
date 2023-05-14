import json

final = []
for dataset in ["KETOD", "Fusedchat"]:
    train = json.load(open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_data\\"+dataset+"\\train.json"))
    val = json.load(open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_data\\"+dataset+"\\val.json"))
    test = json.load(open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_data\\"+dataset+"\\test.json"))
    final = final + train + val + test

inputs = []
for i in range(len(final)):
    inputs.append(final[i]["instruction"].format(
        context=final[i]["context"],
        ontology=final[i]["ontology"] + " || " if final[i]["ontology"] != ""  else "",
        system_action=final[i]["system_action"] + " || " if final[i]["system_action"] != "" else "",
        documents=final[i]["documents"] + " || " if final[i]["documents"] != "" else "",
        style=final[i]["style"]))
with open("C:\ALL\OJT\gradients.baselinev1.dialogstate\data\merged_data\inputs.txt", 'w') as f:
    json.dump(inputs, f, indent=4)