import json
import random
import glob
import os

from src.features.converter import DialConverter
from typing import List, Dict, Union, Optional


class KetodConverter(DialConverter):
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

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruct_path=instruct_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruct_path, ontolopy_path) -> None:
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
        list_instructions = self.define_instruct(instruct_path)
        list_ontologies = self.define_ontology(ontolopy_path)

        data_path_list = glob.glob(os.path.join(self.file_path, '*.json'))
        for data_path in data_path_list:
            filename = os.path.basename(data_path)

            dataset = self.load_datapath(data_path)
            dialogues = self.get_sub_dialogues(dataset)
            # Analyze all dialogues
            list_sample_dict = []
            for dialogue in dialogues:
                instruction = self.get_instruction(list_instructions)
                context = self.get_context(dialogue)
                ontology, map_ontology_domain = self.get_ontology(dialogue, list_ontologies)
                system_action = self.get_system_action(dialogue, list_ontologies)
                documents = self.get_documents(dialogue)
                style = self.get_style('tod') if ontology != "" else self.get_style()
                response = self.get_response(dialogue)

                sample_dict = {
                    "instruction": instruction,
                    "context": context,
                    "ontology": ontology,
                    "system_action": system_action,
                    "documents": documents,
                    "style": style,
                    "response": response,
                    }
                list_sample_dict.append(sample_dict)

            self.save_datapath(list_sample_dict, filename)

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions

    def map_ontology(self, ontologies, domain):
        map_ontology_domain = {}
        count = 0
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count))
            count = count + 1
        return map_ontology_domain

    def get_sub_dialogues(self, dataset):
        sub_dialogues = []
        for i_dialogue in range(len(dataset)):

            # Get all turns in a dialogue
            turns = dataset[i_dialogue]["turns"]

            # Scan all turns in the dialogue
            for i_turn in range(len(turns)):
                turn = turns[i_turn]

                # Separate one sample from first turn to each system turn
                if turn["speaker"] == "SYSTEM":
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
                    if i_turn - self.window_context < 0:
                        sub_dialogues.append(turns[0:i_turn + 1])
                    else:
                        sub_dialogues.append(turns[i_turn - self.window_context:i_turn + 1])
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
            utterance = turn["enriched_utter"] if turn["enrich"] == True else turn["utterance"]
            if turn["speaker"] == "USER":
                context = context + self.tag_speaker + ": " + utterance + " "
            elif turn["speaker"] == "SYSTEM":
                context = context + self.tag_agent + ": " + utterance + " "
        return context.strip()

    def get_documents(self, dialogue):
        documents = ""
        if dialogue[-1]["enrich"] == True:
            documents_list = dialogue[-1]["kg_snippets_text"]
            # Ex: < tag_domain > TRAVEL(...), HOTEL(...) < tag_knowledge > text
            documents = ";".join([doc for doc in documents_list])
        return documents

    def get_ontology(self, dialogue, ontologies):
        # All frames of user
        frames = dialogue[-1]["frames"]
        # Scan all frames_user
        for frame in frames:
            # get domain (domain == service)
            domain = frame["service"].strip().lower()
            map_ontology_domain = self.map_ontology(ontologies, domain)
            ontology = domain.upper() + ":("
            for slot, description in ontologies[domain].items():
                ontology = ontology + map_ontology_domain[slot] + "=" + description + "; "
            ontology = ontology[:-2] + ") || "
        return ontology[:-4], map_ontology_domain

    def get_system_action(self, dialogue, ontologies):
        """
            actions_list = {"OFFER": [slot=value, slot=value, ....],
                            "INFORM": [slot=value, slot=value, ....],
                            .....,
                            "REQUEST": [slot, ....]}
        """

        # List of actions
        frame = dialogue[-1]["frames"][0]
        domain = frame["service"].strip().lower()
        map_ontology_domain = self.map_ontology(ontologies, domain)

        list_action = []
        for action in frame['actions']:
            # slot & value: inform, confirm, offer, offer_intent, inform_count
            if len(action['slot']) > 0 and len(action['values']) > 0:
                if action['act'].lower() in ['offer_intent', 'inform_count']:
                    tmp = str(action['values'][0])
                else:
                    tmp = map_ontology_domain[action['slot']] + '=' + action['values'][0]
                list_action.append(action['act'].lower() + ':(' + tmp + ')')
            # slot: request
            elif len(action['slot']) > 0 and len(action['values']) == 0:
                list_action.append(action['act'].lower() + ':(' + map_ontology_domain[action['slot']] + ')')
            else:  # slot & value empty
                list_action.append(action['act'].lower())

        sample = ' and '.join(text for text in list_action)

        return sample

    def get_style(self, type: str = 'None'):
        if type == 'tod':
            return self.style_tod[random.randint(0, len(self.style_tod) - 1)]
        return self.style_odd[random.randint(0, len(self.style_odd) - 1)]

    def get_response(self, dialogue):
        response = dialogue[-1]["enriched_utter"] if dialogue[-1]["enrich"] == True else dialogue[-1]["utterance"]
        return response

if __name__ == '__main__':

    # ketod_converter = KetodConverter(
    #     file_path='/content/raw/KETOD',
    #     save_path='/content/output/KETOD',
    #     window_context=5).__call__(
    #     instruct_path='./data/instructions_module_3.txt',
    #     ontolopy_path='./data/processed_schema/schema_final.json')
    ketod_converter = KetodConverter(
        file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\\raw\KETOD',
        save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\KETOD',
        window_context=5).__call__(
        instruct_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions\instruct_GradRes.txt',
        ontolopy_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\schema\processed_schema\schema_guided.json')