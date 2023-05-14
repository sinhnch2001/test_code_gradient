from src.config import ontology

path_schema_guided = ""
EOT_SEP = "[EOT]. "
USER_SEP = "USER: "
SYSTEM_SEP = "SYSTEM: "
LIST_RULE = "1. Seek, 2. Chitchat, 3. Database:"

INSTRUCTION1 = f"Instruction: In this task given a dialogue as context you must be given the type of belief state " \
               f"between specified people or speakers. [CTX] <DIALOGUE_CONTEXT> [EOD]. " \
               f"[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state? "
INSTRUCTION2 = f'Instruction: Please use this dialogue [CTX] <DIALOGUE_CONTEXT> [EOD] to predict the type of ' \
               f'belief state between two people. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state?'
INSTRUCTION3 = f'Instruction: This dialogue [CTX] <DIALOGUE_CONTEXT> [EOD] is used to generate ' \
               f'the belief state between two speakers. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state? '
INSTRUCTION4 = f'Instruction: You must be given the type of belief state between specified people or speakers ' \
               f'base on this dialogue [CTX] <DIALOGUE_CONTEXT> [EOD]. '\
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state? '
INSTRUCTION5 = f'Instruction: Please predict the type of belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state? '
INSTRUCTION6 = f'Instruction: Let give the belief state between two persons or two speakers base on ' \
               f'this dialogue [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state? '
INSTRUCTION7 = f"Instruction: Given a dialogue history of two specified speakers as context, " \
               f"identify the type of belief state between them. [CTX] <DIALOGUE_CONTEXT> [EOD]. " \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state ' \
               f'between the specified speakers ?'
INSTRUCTION8 = f"Instruction: Identify the type of belief state between two specified speakers " \
               f"based on the following conversation context [CTX] <DIALOGUE_CONTEXT> [EOD]. " \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the type of belief ' \
               f'state that the specified speaker has ?'
INSTRUCTION9 = f'Instruction: Given a dialogue history as context, identify the belief state over the course of the ' \
               f'conversation. [CTX] <DIALOGUE_CONTEXT> [EOD]. [OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; ' \
               f'Slots: <SLOT> [Q] What is the current belief state based on the above context ? '
INSTRUCTION10 = f'Instruction: Determine the speaker goal and action based on the dialogue history between specified ' \
                f'people or speakers as context. [CTX] <DIALOGUE_CONTEXT> [EOD]. [OPT] {ontology.LIST_USER_ACT}; ' \
                f'{LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the goal and action ?'
INSTRUCTION11 = f'Instruction: Based on the following conversation context [CTX] <DIALOGUE_CONTEXT> [EOD],' \
                f' generate a belief state of the speaker using the following options' \
                f' [OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT> [Q] What is the belief state ? '
INSTRUCTION12 = f'Instruction: By examining the given conversation, you must be able to recognize the belief state that exists  ,' \
               f'between the two individuals. [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT>[Q] What is the belief state? '
INSTRUCTION13 = f'Instruction: You will be required to analyze the conversation provided and determine the type of belief state between  ,' \
               f'the specified individuals or speakers. [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT>[Q] What is the belief state? '
INSTRUCTION14 = f'Instruction: The goal of this assignment is to determine the belief state between the specified individuals by analyzing ,' \
               f'the dialogue provided. [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT>[Q] What is the belief state? '
INSTRUCTION15 = f'Instruction: Your objective is to identify the belief state between two people or speakers by analyzing ,' \
               f'the given dialogue. [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT>[Q] What is the belief state? '
INSTRUCTION16 = f'Instruction: You need to determine the type of belief state between the specified speakers based on,' \
               f'this provided dialogue. [CTX] <DIALOGUE_CONTEXT> [EOD]. ' \
               f'[OPT] {ontology.LIST_USER_ACT}; {LIST_RULE} <DOMAIN>; Slots: <SLOT>[Q] What is the belief state? '
