import json
import glob
import os

from typing import List, Dict, Union, Optional
from src.features.converter import DialConverter


class KetodConverter(DialConverter):
    def __init__(self,
                 file_path: str,
                 save_path: str,
                 tag_speaker: str = 'USER',
                 tag_agent: str = 'AGENT',
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
        self.useracts = ['INFORM', 'REQUEST', 'INFORM_INTENT', 'NEGATE_INTENT',
                         'AFFIRM_INTENT', 'AFFIRM', 'NEGATE', 'SELECT', 'THANK_YOU',
                         'GOODBYE', 'GREET', 'GENERAL_ASKING', 'REQUEST_ALTS']

    def __call__(self, instruct_path, ontolopy_path):
        print(f"Start  processing {self.__class__.__name__}")
        self.process(instruct_path=instruct_path, ontolopy_path=ontolopy_path)
        print(f"Finish processing {self.__class__.__name__} at {self.save_path}")

    def process(self, instruct_path, ontolopy_path) -> None:
        # Separate complete dialogues to sub dialogues
        list_instructions = self.define_instruct(instruct_path)
        list_ontologies = self.define_ontology(ontolopy_path)

        data_path_list = glob.glob(os.path.join(self.file_path, '*.json'))
        for data_path in data_path_list:
            filename = os.path.basename(data_path)

            dataset = self.load_datapath(data_path)
            dialogues = self.get_sub_dialouge(dataset)

            # Analyze all dialogues
            sample = []
            for subdiag in dialogues:
                item = dict()
                list_turn = []

                # input
                for utter in subdiag:
                    if utter['speaker'] == "USER":
                        speaker = "USER: "
                    else:
                        speaker = "AGENT: "
                    list_turn.append(speaker + utter['utterance'])
                if len(list_turn) == 1:
                    item['context'] = ''
                    item['current_query'] = str(list_turn)
                else:
                    item['context'] = ' '.join([list_turn[idx].strip() for idx in range(len(list_turn) - 1)])
                    item['current_query'] = str(list_turn[-1][6:])

                item['instruction'] = list_instructions[0]
                item['list_user_action'] = ', '.join([item.lower() for item in self.useracts])
                # label
                ontologies_dm, onto_mapping_ls = self.get_ontology(subdiag, list_ontologies)
                item['ontology'] = '||'.join(dm for dm in ontologies_dm) if len(ontologies_dm) > 1 else ontologies_dm[0]

                frm = []
                for idx, frame in enumerate(subdiag[-1]["frames"]):
                    list_action = []
                    domain = frame["service"].strip().upper()
                    for action in frame['actions']:
                        # Inform
                        if len(action['slot']) > 0 and len(action['values']) > 0:
                            if action['act'].lower() in ['affirm_intent', 'inform_intent']:
                                tmp = frame['state']['active_intent']  # get current state
                            else:
                                tmp = onto_mapping_ls[idx][action['slot']] + '=' + action['values'][0]
                            list_action.append(action['act'].lower() + ':(' + tmp + ')')
                        # request
                        elif len(action['slot']) > 0 and len(action['values']) == 0:
                            list_action.append(
                                action['act'].lower() + ':(' + onto_mapping_ls[idx][action['slot']] + ')')
                        # TOD: Goodbye - thanks - greeting - request_alts - affirm - negate - negate_intent - select
                        # ODD: General_asking | Goodbye - thanks -greeting
                        else:
                            list_action.append(action['act'].lower())

                    frm.append(domain + ':[' + ' and '.join(text for text in list_action) + ']')
                item['label'] = ' || '.join(item for item in frm) if len(frm) > 0 else frm[0]

                sample.append(item)

            self.save_datapath(sample, filename)

    def get_sub_dialouge(self, dataset):
        """
        This function is to get utterance (<=5 utterance)
        :return: the list of list, each list contain utterances for each Input
                EX: [[utterance1, utterance2, utterance3, ...],
                    [utterance4, utterance5, utterance6, ...]]
        """
        list_utter = []
        cw = self.window_context - 1  # slide window
        for dialogue in dataset:
            list_turns = dialogue['turns']
            for idx in range(len(list_turns)):
                if idx % 2 == 0:  # current_utter is user
                    child_dialogue = list_turns[max(0, idx - cw):max(0, idx + 1)]
                    list_utter.append(child_dialogue)
        return list_utter

    def map_ontology(self, ontologies, domain, count=0):
        map_ontology_domain = {}
        for slot in ontologies[domain].keys():
            map_ontology_domain.setdefault(slot, "slot" + str(count))
            count = count + 1
        return map_ontology_domain

    def get_ontology(self, dialogue, ontologies):
        results, onto_mapping_ls = [], []
        for idx, frame in enumerate(dialogue[-1]["frames"]):
            # get domain (domain == service)
            domain = frame["service"].strip().lower()
            onto_mapping = self.map_ontology(ontologies, domain)
            tmps = [onto_mapping[key] + "=" + value
                    for key, value in ontologies[domain].items()]
            value_onto = domain.upper() + ":(" + '; '.join(tmp for tmp in tmps) + ")"

            results.append(value_onto)
            onto_mapping_ls.append(onto_mapping)

        return results, onto_mapping_ls

    def define_ontology(self, ontolopy_path: Optional[str] = None) -> Union[Dict, List[Dict]]:
        with open(ontolopy_path) as f:
            ontologies = json.load(f)
        return ontologies

    def define_instruct(self, instruct_path) -> List[str]:
        with open(instruct_path) as f:
            instructions = f.readlines()
        return instructions

    def save_datapath(self, data_processed: List[Dict], filename: str):
        with open(os.path.join(self.save_path, filename), 'w') as f:
            json.dump(data_processed, f, indent=4)

    def load_datapath(self, data_path) -> List[Dict]:
        with open(data_path, 'r+') as f:
            dataset = json.load(f)
        return dataset


if __name__ == '__main__':
    converter = KetodConverter(file_path='/content/raw/KETOD',
                               save_path='/content/interim/GradSearch/KETOD',
                               window_context=5)
    converter.__call__(instruct_path='./data/instruct_GradSearch.txt',
                       ontolopy_path='./data/schema_guided.json')