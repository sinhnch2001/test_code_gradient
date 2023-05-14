import abc
from typing import Optional, List, Union, Dict, Tuple
import pandas as pd


class AbstractCoverter(metaclass=abc.ABCMeta):

    def __init__(self, file_path, save_path) -> None:
        self.file_path = file_path
        self.save_path = save_path

    @abc.abstractmethod
    def load_datapath(self):
        """
        Read the data file (.json, .csv, ...)
        :return: the list of dictionaries or the dictionary of samples in the dataset
        """
        pass

    @abc.abstractmethod
    def save_datapath(self):
        """
        Save the sample after processing raw input
        :return: list of dictionaries
        """
        pass

    @abc.abstractmethod
    def process(self):
        """
        Implement your convert logics in this function
        """
        raise NotImplementError


class DialConverter(AbstractCoverter):
    def __init__(
            self,
            file_path: str,
            save_path:str,
            tag_speaker:str = 'USER',
            tag_agent:str = 'AGENT',
            window_context: int = 0):
        super().__init__(file_path, save_path)
        self.tag_speaker = tag_speaker
        self.tag_agent = tag_agent
        self.window_context = window_context if window_context > -1 else -1

    @abc.abstractmethod
    def define_instruct(
            self,
            instruct_path: Optional[str] = None,
    ) -> List[str]:
        """ Define or load instruction template

        Args:
            instruct_path: where on disk the source instruct data is located.

        Returns:
            List sample instruction
        """
        pass

    @abc.abstractmethod
    def define_ontology(
            self,
            ontolopy_path: Optional[str] = None
    ) -> Union[Dict, List[Dict]]:
        """Define and load an ontology use in Task oriented Domain Dialogue
        for inference or evaluate

        Args:
            ontolopy_path: where on disk the source ontology template is located.

        Returns:
            The list of dictionaries or the dictionary of samples in the dataset
        """

        pass

    def load_schema(
            self,
            schema_path:Optional[str] = None,
    )-> Union[Tuple[Dict, Dict], None]:
        """
        Read schema guided to map old domain to new domain (training ToD)
        :return: two dictionaries to map old_domain to new_domain, old_slot to new_slot
        """
        if schema_path:
            # Read schema guided file
            schema_guided = pd.read_excel(schema_path, None)
            # Create dataframe of schema_guided
            df_schema = pd.DataFrame(columns=['domain', 'old slots', 'original dataset', 'new slots'])
            for schema in schema_guided.values():
                schema.columns = schema.columns.str.lower()
                schema.columns = schema.columns.str.replace('original domain', 'original dataset')
                df_schema = pd.concat([df_schema, schema], axis=0, ignore_index=True)
                df_schema['original dataset'] = df_schema['original dataset'].str.lower()
                df_schema['original dataset'] = df_schema['original dataset'].str.strip()
                df_schema['old slots'] = df_schema['old slots'].str.strip()
                df_schema['new slots'] = df_schema['new slots'].str.strip()

            df_schema = df_schema.dropna(how='all').reset_index(drop=True)
            df_schema = df_schema.fillna(method='ffill')
            mask = df_schema['original dataset'] == 'fused chat'
            df_schema.loc[mask, 'original dataset'] = df_schema.loc[mask, 'domain']
            # Create dict to map old domain to new domain
            dict_domain = dict(zip(df_schema['original dataset'], df_schema['domain']))

            dict_slot = dict()
            for new_domain in set(dict_domain.values()):
                child_df = df_schema[df_schema['domain'] == new_domain]
                dict_slot[new_domain] = dict(zip(child_df['old slots'], child_df['new slots']))

            return dict_domain, dict_slot

        return None