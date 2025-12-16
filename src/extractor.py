from typing import List

from pydantic import BaseModel, Field
from trustcall import create_extractor

from src.llm import model


class Action(BaseModel):
    """
    Represents an interactive feature on the webpage that a human can perform.
    An action is an interactive UI element or a sequence of related elements (like a form) 
    that, when engaged, leads to a specific change or outcome (e.g., login, search, navigation).
    All related inputs and buttons for a single goal MUST be grouped into ONE Action.
    """
    description: str = Field(
        description='A short, human-readable summary of the **complete goal** of the action (e.g., "Sign In to Netflix", "Perform a Web Search").'
    )
    process: str = Field(
        description='''
            A detailed, sequential, step-by-step description of **all** human interactions required to complete the action, 
            including filling inputs, toggling checkboxes, and finally clicking the submit button.
            (e.g., "enter email, enter password, check 'Remember me' checkbox, then click the Sign In button").
        '''
    )

class Info(BaseModel):
    """
    Represents a read-only piece of information provided by the webpage.
    This content is purely for the user's viewing and cannot be interacted with 
    to achieve a result. Elements used in *any* Action must be strictly excluded from this list.
    """
    description: str = Field(
        description='''
            A comprehensive, human-readable sentence summarizing related static data points or text. 
            Crucially, related items (e.g., a metric value, its label, and its percentage change) 
            MUST be combined into a single, cohesive, descriptive statement. 
            Do NOT list individual text fragments.
            Example: "The Total Views metric is $3,456K, showing an increase of 0.43%."
        '''
    )

class Features(BaseModel):
    """
    The complete set of features provided by the rendered HTML page.
    This includes both interactive Actions and read-only Information.
    """
    actions: List[Action] = Field(
        description='''
            A list of all interactive actions available on the rendered webpage. 
            All components of a single form or goal must be combined into one action.
        '''
    )
    info_: List[Info] = Field(
        description='''
            A list of all read-only information displayed on the rendered webpage. 
            **MUST NOT** include any element or text that is part of an Action.
        '''
    )

bmodel = create_extractor(
    llm=model,
    tools=[Features],
    tool_choice='Features'
)

# Created By Amit Mahapatra