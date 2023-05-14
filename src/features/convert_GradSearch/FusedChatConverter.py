from src.features.converter import DialConverter
import json
from typing import List, Dict, Tuple, Union, Optional
import json
import random

class FusedchatConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
                 style_tod: List[str] = ["politely"],
                 style_odd: List[str] = ["empathically", "safety", "friendly"],
                 window_context: int = 0,
                 ) -> None:
        """
        Args:
            save_path: path to save the processed dataset
        """
        super().__init__(file_path,
                         save_path,
                         tag_speaker,
                         tag_agent,
                         window_context,
                         )
        self.style_tod = style_tod
        self.style_odd = style_odd

    def load_datapath(self) -> List[Dict]:
        with open(self.file_path + '\\train.json') as f:
            train = json.load(f)
        with open(self.file_path + '\\val.json') as f:
            val = json.load(f)
        with open(self.file_path + '\\test.json') as f:
            test = json.load(f)

        dataset = train + val + test
        return dataset

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def map_ontology(self, ontologies, list_domains):
        map_ontology_domain = {}
        count = 0
        for domain in list_domains:
            map_slot = {}
            for slot in ontologies[domain.lower().strip()].keys():
                map_slot.setdefault(slot, "slot" + str(count))
                map_ontology_domain.setdefault(domain, map_slot)
                count = count + 1
        return map_ontology_domain

    def get_sub_dialogues(self, dataset):
        sub_dialogues = []
        for i_dialogue in range(len(dataset)):

            # Get all log in a dialogue
            log = dataset[i_dialogue]["log"]
            [log[j].update({"speaker": "USER"} if j % 2 == 0 else {"speaker": "SYSTEM"}) for j in range(len(log))]
            dialog_action = dataset[i_dialogue]["dialog_action"]
            [log[int(j)].update(dialog_action[j]) for j in dialog_action]

            for i_log in range(len(log)):
                if log[i_log]["dialog_act"] != {}:
                    for dialog_act in list(log[i_log]["dialog_act"].keys()):
                        if "general" in dialog_act:
                            del log[i_log]["dialog_act"][dialog_act]

                if log[i_log]["dialog_act"] == {}:
                    if i_log==0:
                        dict_domain_chitchat = {}
                        for dialog_act in list(log[i_log+1]["dialog_act"].keys()):
                            domain, action = dialog_act.split('-')
                            dict_domain_chitchat.setdefault(domain + "-chitchat", None)
                        log[i_log]["dialog_act"] = dict_domain_chitchat

                    if log[i_log-1]["dialog_act"] == {"chitchat": []}:
                        log[i_log]["dialog_act"] = {"chitchat": []}
                    else:
                        dict_domain_chitchat = {}
                        for dialog_act in list(log[i_log-1]["dialog_act"].keys()):
                            domain, action = dialog_act.split('-')
                            dict_domain_chitchat.setdefault(domain+"-chitchat",None)
                        log[i_log]["dialog_act"] = dict_domain_chitchat

            if log[0]["dialog_act"] == {"chitchat": []}:
                for i_log in range(len(log)):
                    if log[i_log]["dialog_act"] != {"chitchat": []}:
                        dict_domain_chitchat = {}
                        for dialog_act in list(log[i_log]["dialog_act"].keys()):
                            domain, action = dialog_act.split('-')
                            dict_domain_chitchat.setdefault(domain + "-chitchat", None)
                        for j_log in range(i_log):
                            log[j_log]["dialog_act"] = dict_domain_chitchat
                        break
            else:
                for i_log in range(len(log)):
                    if log[i_log]["dialog_act"] == {"chitchat": []}:
                        dict_domain_chitchat = {}
                        for dialog_act in list(log[i_log-1]["dialog_act"].keys()):
                            domain, action = dialog_act.split('-')
                            dict_domain_chitchat.setdefault(domain + "-chitchat", None)
                        for j_log in range(i_log,-1):
                            log[j_log]["dialog_act"] = dict_domain_chitchat
                        break
            # Scan all log in the dialogue
            for i_log in range(len(log)):

                # Separate one sample from first turn to each system turn
                if i_log % 2 == 0:
                    """
                        if self.num_of_utterances == 5
                            EX 0: i_turn = 3
                            i_turn - self.num_of_utterances = -2
                            sub_dialogues.append(turns[0:4]) context: 0,1,2 response: 3 

                            EX 1: i_turn = 5
                            i_turn - self.num_of_utterances = 0
                            sub_dialogues.append(turns[0:6]) context: 0,...,4 response: 5

                            EX 2: i_turn = 7
                            i_turn - self.num_of_utterances = 2
                            sub_dialogues.append(turns[2:8]) context: 2,...,6 response: 7
                    """
                    if i_log - self.window_context < 0:
                        sub_dialogues.append(log[0:i_log + 1])
                    else:
                        sub_dialogues.append(log[i_log - self.window_context:i_log + 1])
        return sub_dialogues

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction[:-1]

    def get_context(self, dialogue):
        context = ""
        """
            Concat text with length for context
            Ex: context = <tag_speaker> .... <tag_agent> .... <tag_speaker>
        """
        for i in range(len(dialogue) - 1):
            turn = dialogue[i]
            if turn["speaker"] == "USER":
                context = context + self.tag_speaker + ": " + turn["text"] + " "
            elif turn["speaker"] == "SYSTEM":
                context = context + self.tag_agent + ": " + turn["text"] + " "
        return context.strip()

    def get_query(self, dialogue):
        query = dialogue[-1]["text"]
        return query

    def get_ontology(self, dialogue, ontologies):
        ontology = ""
        list_domain_dialogue = []
        dialog_act = dialogue[-1]["dialog_act"]
        for domain_action, list_slots in dialog_act.items():
            if domain_action != "chitchat":
                domain, action = domain_action.split('-')
                if domain.lower().strip() != "general":
                    if domain.lower().strip() not in list_domain_dialogue:
                        list_domain_dialogue.append(domain.lower().strip())

        map_ontology_domain = self.map_ontology(ontologies, list_domain_dialogue)
        for domain in list_domain_dialogue:
            ontology = ontology + domain.upper() + ":("
            for slot, description in ontologies[domain].items():
                ontology = ontology + map_ontology_domain[domain][slot] + "=" + description + ";"
            ontology = ontology[:-1] + ") || "
        return ontology[:-4]

    def get_label(self, child_dialogue, ontology):
        dict_input = {}
        dict_ontology = dict()
        map_onto = dict()
        list_domain = dict()
        dict_ontology_domains = dict()
        if 'dialog_action' not in child_dialogue[-1].keys():
            dict_ontology= {}
        else:
            frame = child_dialogue[-1]['dialog_action']["dialog_act"]
            service = [i for i in frame]
            domain_set = set(i.split("-")[0] for i in service)

            for i in domain_set:
                i = i.lower()
                if i in ontology.keys():
                    dict_ontology_domain = self.map_ontology(ontology, i)
                    dict_ontology_domains[i] = dict_ontology_domain
                    for k,v in ontology[i].items():
                        map_onto[dict_ontology_domain[k]] = v

                    dict_ontology[i] = map_onto

        if 'dialog_action' not in child_dialogue[-1].keys():
            dict_input['output'] = "Unsure about answer, you should find with SearchEngine [general]"
        # if current utterance has domain general : label = Chitchat: action
        elif 'general' in domain_set:
            dict_input['output'] = "Unsure about answer, you should find with SearchEngine [general]"
        # if current utterance is TOD
        else:
            domain_actions = []
            domain_actions.append(child_dialogue[-1]['dialog_action']['dialog_act'])
            for domain_action in domain_actions:
                for domain_key, domain_val in domain_action.items():
                    domain1 = domain_key.split("-")[0]
                    action1 = domain_key.split("-")[1].lower()
                    action_slot = dict()
                    action_slot[action1] = domain_val

                    for i in range(0,len(domain_val)):
                        if domain_val[i][0] == 'none':
                            domain_val = [" "]
                            action_slot[action1] = domain_val
                        else:
                            a = dict_ontology_domains[domain1.lower()][domain_val[i][0].lower()]
                            domain_val[i][0] = a
                            action_slot[action1] = domain_val
                    if domain1 not in list_domain.keys():
                        list_domain[domain1] = action_slot
                    else:
                        list_domain[domain1].__setitem__(action1, domain_val)

            output = dict()
            for k, v in list_domain.items():
                list_ac = []
                for ke, va in v.items():
                    list_slot = ["=".join(val) for val in va if val]
                    if list_slot:
                        c = ke + ": " + "(" + ";".join(list_slot) + ")"
                        list_ac.append(c)
                list_ac = [x for x in list_ac if "()" not in x]
                if list_ac:
                    f = "[" + ", ".join(list_ac) + "]"
                    output[k] = f

            none_list = [k for k, v in output.items() if v == '[]']
            for zeros in none_list:
                del output[zeros]
            if len(output) == 1:
                k, v = next(iter(output.items()))
                dict_input['output'] = f"{k}={v}"
            elif len(output) == 0:
                dict_input['output'] = ''
            else:
                complete_output = [f"{k}={v}" for k, v in output.items()]
                dict_input['output'] = ";".join(complete_output)
        dict_input['output'] = dict_input['output'].replace(": ( )", "")
        return dict_input['output'], dict_ontology

    def process(self, dataset: List[Dict], instruct_path, ontolopy_path) -> None:
        """Implement your convert logics in this function
            1. `instruction`: instruction for zero-shot learning
            2. `context`: means dialogue history
            3. `state_of_user`: a support document to response current
                user's utterance
            4. `system_action`: what system intends to do
            5. `response`: label for response generation
        A dataset after being formatted should be a List[Dict] and saved
        as a .json file
        Note: text preprocessing needs to be performed during transition
                from crawl dataset to module3-based format
        """
        # Separate complete dialogues to sub dialogues
        dialogues = self.get_sub_dialogues(dataset)

        # Load domain_map, slot_map and instructions
        list_instructions = self.define_instruct(instruct_path)
        list_ontologies = self.define_ontology(ontolopy_path)

        # Analyze all dialogues
        list_sample_dict = []
        for dialogue in dialogues:
            instruction = self.get_instruction(list_instructions)
            context = self.get_context(dialogue)
            ontology = self.get_ontology(dialogue, list_ontologies)
            query = self.get_query(dialogue)
            # label = self.get_label(dialogue, list_ontologies)

            sample_dict = {
                "instruction": instruction,
                "context": context,
                "query": query,
                "ontology": ontology,
                "label": "label",
            }
            list_sample_dict.append(sample_dict)

        return list_sample_dict

    def save_datapath(self, data_train: List[Dict], data_val: List[Dict], data_test: List[Dict]):
        with open(self.save_path + "\\train.json", 'w') as f:
            json.dump(data_train, f, indent=4)
        with open(self.save_path + "\\val.json", 'w') as f:
            json.dump(data_val, f, indent=4)
        with open(self.save_path + "\\test.json", 'w') as f:
            json.dump(data_test, f, indent=4)

    def get_classified_data(self, data_processed: List[Dict], knowledge_tag: str) -> Tuple[
        List[Dict], List[Dict], List[Dict]]:
        """Clasify dataset into chitchat, qa, tod datasets
        Args:
            dataset (List[Dict]): full processed data

        Returns:
            chitchat (List[Dict]): chitchat dataset
            qa (List[Dict]): QA dataset
            tod (List[Dict]): TOD dataset
        """
        chitchat = []
        qa = []
        tod = []
        for sample in data_processed:
            if sample['ontology'] == 'NONE':
                chitchat.append(sample)
            elif knowledge_tag in sample['ontology']:
                qa.append(sample)
            else:
                tod.append(sample)
        return chitchat, qa, tod

    def distribute_datsets(self, chitchat: List[Dict], qa: List[Dict], tod: List[Dict], valid_ratio: float,
                           test_ratio: float) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Distribute data to train, val, test sets by type and ratio
        Args:
            chitchat (List[Dict]): chitchat dataset
            qa (List[Dict]): QA dataset
            tod (List[Dict]): TOD dataset
            valid_ratio (float): ratio of splitting valid set
            test_ratio (float): ratio of splitting test set
        Returns:
            train (List[Dict]): train dataset
            valid (List[Dict]): valid dataset
            test (List[Dict]): test dataset
        """
        train = []
        valid = []
        test = []
        train_ratio = 1.0 - valid_ratio - test_ratio
        key_data = {'chitchat': chitchat, 'qa': qa, 'tod': tod}
        for dataset in key_data.values():
            train_size = round(len(dataset) * train_ratio)
            valid_size = round(len(dataset) * (train_ratio + valid_ratio))
            train += dataset[:train_size]
            valid += dataset[train_size:valid_size]
            test += dataset[valid_size:]

        return train, valid, test

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")

        dataset = self.load_datapath()
        data_processed = self.process(
            dataset=dataset,
            instruct_path=instruct_path,
            ontolopy_path=ontolopy_path)
        chitchat, qa, tod = self.get_classified_data(
            data_processed=data_processed,
            knowledge_tag="supported documnets")
        data_train, data_val, data_test = self.distribute_datsets(chitchat, qa, tod, 0.005, 0.005)
        self.save_datapath(data_train, data_val, data_test)

        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

fusedchat_converter = FusedchatConverter(
    file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\\raw\Fusedchat',
    save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_data\Fusedchat',
    window_context=4)
data = fusedchat_converter.load_datapath()
a = fusedchat_converter.get_sub_dialogues(data)
# context_list = []
# for i in a:
#     context_list.append(fusedchat_converter.get_query(i))
with open("/data/interim/GradSearch\Fusedchat\sub_dialogue.json", 'w') as f:
    json.dump(a, f, indent=4)

# fusedchat_converter = FusedchatConverter(
#     file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\\raw\Fusedchat',
#     save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\interim\GradSearch\Fusedchat',
#     window_context=4).__call__(
#     instruct_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions_module_1.txt',
#     ontolopy_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\processed_schema\schema_final.json')



