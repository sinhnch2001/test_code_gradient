import json
import random
import glob
import os

from src.features.converter import DialConverter
from typing import List, Dict, Union, Optional

WOZ_DMS = ['taxi', 'police', 'hospital', 'hotel',
           'attraction', 'train', 'restaurant']


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
        self.window_context = window_context

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruct_path=instruct_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruct_path, ontolopy_path):
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
        data_path_list = glob.glob(os.path.join(self.file_path, '*.json'))
        list_instructions = self.define_instruct(instruct_path)
        list_ontologies = self.define_ontology(ontolopy_path)
        for data_path in data_path_list:
            filename = os.path.basename(data_path)
            dataset = self.load_datapath(data_path)
            self.process_datasets(dataset)
            samples = []
            # Analyze all dialogues
            for dialogue in dataset:
                # get summarize dialogue
                gold_domain = self.get_gold_dm(dialogue, list_ontologies)
                # process dialogue into sub-dialogue
                cw = self.window_context - 1

                for idx in range(len(dialogue['log'])):
                    sub_dialogue = []
                    if idx % 2 != 0:
                        sub_dialogue.append(dialogue['log'][max(0, idx - cw):max(0, idx + 1)])
                        sample = self.get_sample(sub_dialogue, gold_domain,
                                                 list_ontologies, list_instructions)
                        samples.append(sample)

            self.save_datapath(samples, filename)

    def get_sample(self, sub_dialogue, gold_dm, list_ontologies, list_instructions):
        for _, child in enumerate(sub_dialogue):
            # get context
            item = dict()
            ls_turn = []
            for turn in child:
                speaker = 'USER: ' if turn['speaker'] == 'USER' else 'SYSTEM: '
                ls_turn.append(speaker + turn['text'])

            tmp = 0 if len(ls_turn) == 1 else 1

            item['context'] = ' '.join([ls_turn[idx].strip()
                                        for idx in range(len(ls_turn) - tmp)])
            item['instruction'] = self.get_instruction(list_instructions)
            item['label'] = child[-1]['text']
            item['documents'] = ''  # fushedchat - no knowledge

            # system_action
            ls_sysacts = []
            for domain_action, frames in child[-1]['dialog_act'].items():
                """
                domain_action: domain | action
                frame: list[[slot, value], [slot, value]]

                """
                dm_act = domain_action.split('-')

                if dm_act[0] == 'chitchat':  # ODD
                    item['system_action'] = ''
                    item['ontology'] = ' || '.join(idx for idx in gold_dm)
                    item['style'] = self.get_style('odd')
                else:  # TOD
                    if dm_act[0].lower() in WOZ_DMS:
                        idx_name = dm_act[0].lower()
                        value_onto, onto_mapping = self.get_ontology(idx_name, list_ontologies)
                        item['ontology'] = value_onto

                        ls_txt = []
                        for value in frames:
                            if value[0] in ['choice', 'none']:
                                ls_txt.append(value[1])
                            else:
                                ls_txt.append(onto_mapping[value[0]] + '=' + value[1])

                        sysacts = [dm_act[1] + '(' + uttr + ')' for uttr in ls_txt]
                        temp_sysacts = ' and '.join(sysact for sysact in sysacts)
                        ls_sysacts.append(temp_sysacts.replace('(none)', '').replace('=?', ''))

                    else:  # Booking | general (general-thank | general-bye) just a state not a domain
                        # Get history action to determine "domain" but system_act with current user
                        # history = history[-2]; system_act = history[current]
                        history_act = list(child[-2]['dialog_act'].keys())

                        if len(history_act) == 0:
                            item['ontology'] = ''
                            item['system_action'] = ''
                        else:
                            dm_name = (history_act[0].split('-')[0]).lower()
                            if dm_name == 'general':  # general-thank, general-bye, general-welcome
                                item['system_action'] = 'reqmore' if dm_act[1] == 'reqmore' else ''
                                item['ontology'] = ''
                            else:
                                value_onto, _ = self.get_ontology(dm_name, list_ontologies)
                                item['ontology'] = value_onto

                                ls_txt = [value[0] + '=' + value[1] for value in frames]
                                sysacts = [dm_act[1] + '(' + uttr + ')' for uttr in ls_txt]
                                temp_sysacts = ' and '.join(sysact for sysact in sysacts)
                                item['system_action'] = temp_sysacts.replace('(none=none)', '') \
                                    .replace('=?', '') \
                                    .replace('NoBook', 'notify_failure') \
                                    .replace('Book:', 'notify_success:') \
                                    .replace('Inform', 'offer_intent')
                    item['style'] = self.get_style('tod')

            item['system_action'] = ' and '.join(item for item in ls_sysacts)

        return item

    def process_datasets(self, datasets):
        for dialogue in datasets:
            for idx, turn in enumerate(dialogue['log']):
                turn['speaker'] = 'USER' if idx % 2 == 0 else 'SYSTEM'
                if 'dialog_act' not in turn.keys():  # TOD original
                    item = dialogue['dialog_action'].get(str(idx))
                    turn['dialog_act'] = item['dialog_act']
                    turn['span_info'] = item['span_info']

        return

    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def map_ontology(self, ontologies, domain, count=0):
        map_ontology_domain = {}
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count))
            count = count + 1
        return map_ontology_domain

    def get_gold_dm(self, dialogue, ontologies):
        gold_dm = []
        for dm, value in dialogue['goal'].items():
            if dm in WOZ_DMS:
                onto_mapping = self.map_ontology(ontologies, dm)
                tmps = [onto_mapping[k] + "=" + v
                        for k, v in ontologies[dm].items()]
                value_onto = dm.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"
                if len(value) != 0:
                    gold_dm.append(value_onto)

        return gold_dm

    def get_ontology(self, domain_name, ontologies):
        onto_mapping = self.map_ontology(ontologies, domain_name)
        tmps = [onto_mapping[key] + "=" + value
                for key, value in ontologies[domain_name].items()]
        value_onto = domain_name.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"

        return value_onto, onto_mapping

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def get_instruction(self, list_instructions):
        random_instruction = list_instructions[random.randint(0, len(list_instructions) - 1)]
        return random_instruction[:-1]

    def get_style(self, type: str = 'None'):
        if type == 'tod':
            return self.style_tod[random.randint(0, len(self.style_tod) - 1)]
        return self.style_odd[random.randint(0, len(self.style_odd) - 1)]


if __name__ == '__main__':
    # TEST
    # fusedchat_converter = FusedchatConverter(
    #     file_path='/content/raw/Fusedchat',
    #     save_path='/content/interim/GradRes/FUSHEDCHAT',
    #     window_context=5).__call__(
    #     instruct_path='data/instruct_GradRes.txt',
    #     ontolopy_path='data/schema_guided.json')
    fusedchat_converter = FusedchatConverter(
        file_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\\raw\FUSEDCHAT',
        save_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\datasets\interim\GradRes\FUSEDCHAT',
        window_context=5).__call__(
        instruct_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\instructions\instruct_GradRes.txt',
        ontolopy_path='C:\ALL\OJT\gradients.baselinev1.dialogstate\data\schema\processed_schema\schema_guided.json')