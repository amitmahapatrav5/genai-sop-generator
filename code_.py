from typing import List
import os

from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from trustcall import create_extractor
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
# with open("dashboard.html") as f:
#     content = f.read()

# tokens = tokenizer.encode(content)

model = ChatOllama(
    base_url='https://ollama.com',
    model='gpt-oss:120b',
    lc_secrets={
        'headers' : {'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')}
    }
)

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

result = bmodel.invoke(
    f"""
        Role: UI Feature Analyzer
        Context: You are an expert AI model designed to parse raw HTML content and strictly categorize its visible features as either interactive actions or read-only information, mimicking the perception of a human viewing the rendered web page.
        
        Thought Process (Internal to the LLM):
        1.  **Analyze Actions:** Identify and group related interactive elements (forms, buttons, links) into single logical **Actions**. Ensure the Action's `process` describes the full sequential interaction path.
        2.  **Analyze Information:** Identify all static, read-only text content.
        3.  **Group Information (NEW):** Combine related data points, labels, and statistics into **ONE** cohesive, descriptive statement. For instance, combine "John Doe", "User Profile is", and "Frontend Engineer" into a single summary statement. Combine a metric's label, its value, and its change percentage into one comprehensive entry.
        4.  **Filter Redundancy:** Strictly filter this content: If any text, label, or UI element has been included in the `actions` list (even if it's a placeholder or label), it **MUST BE EXCLUDED** from the `info_` list.
        5.  **Format:** Format the grouped, filtered results into the required `Features` tool call.
        
        Task:
        Analyze the provided raw HTML content to identify all elements that would be visible in a rendered web browser, classifying them as the site's Features.
        1.  **Actions:** Identify all interactive features. All fields, toggles, and buttons belonging to a single goal must be combined into **ONE** Action.
        2.  **Information (info_):** Identify all static, read-only content. **All related data, labels, and metrics must be grouped into a single, comprehensive, human-readable sentence** that summarizes the displayed information (e.g., combining a KPI's value and its label).
        
        Constraints:
        * **STRICTLY UI-BASED:** Base all findings exclusively on what is visible in the rendered UI.
        * **NO INFERENCE:** Do not infer any backend logic or invisible behavior.
        * **GROUPING (ACTIONS):** All components of a single form or interactive goal MUST be grouped into **ONE** single `Action`.
        * **GROUPING (INFO):** Related data points (like a statistic and its label) must be grouped into a **single, descriptive, human-readable sentence** in the `description` field of the `Info` model. Do not list fragments.
        * **REDUNDANCY:** Any text, label, or UI element that is part of an identified `Action` **MUST NOT** be included in the `info_` list.
        * **CRITICAL OUTPUT RULE:** You must format the final analysis using the provided tool/function **Features**. The analysis is useless if not structured this way.
        * **NO OMISSION:** All fields must be present. If no Actions or Information are found, return the corresponding list as empty (`[]`).
        
        Output Format:
        * **MANDATORY TOOL CALL:** You MUST output **ONLY** a single tool call to the **Features** function.
        * **ABSOLUTE RULE:** Absolutely no commentary, reasoning, markdown, or text **before, after, or outside** the `Features` tool call. This is a machine-readable requirement.
        * The output structure must exactly match the `Features` schema.
        
        Examples:
        HTML Content Snippet: (Dashboard Example)
        
        Example Action (Grouped):
          - description: "Filter Dashboard Data by Date Range"
          - process: "Click the dropdown or input field showing '12.04.2023 - 12.05.2024' to open the date picker, select a new start and end date, and click 'Apply'."
        
        Example Information (Grouped and Human-Readable):
          - description: "The current user is John Doe, whose profile title is Frontend Engineer."
          - description: "The Dashboard Overview shows Total Views are $3,456K, which is up 0.43%."
          - description: "The Total Sales metric for the period 12.04.2023 - 12.05.2024 is currently displayed."
          - description: "The company's brand or product name displayed in the header is Zenith (Z)."
        
        HTML CONTENT:
        {content}
    """
)

features = result['responses'][0] if result['responses'] else None

if features:
    print('\nAction\n')
    for action in features.actions:
        print(action)
    print('\nInformation\n')
    for info in features.info_:
        print(info)
else:
    print('Model did not perform tool call. Please Retry.')


# Action

# description='Toggle the sidebar visibility' process='Click the menu icon button (three bars) in the header to open or close the sidebar.'
# description='Close the sidebar' process="Click the 'X' button at the top right of the sidebar to hide it."
# description='Navigate to the home page via the brand link' process="Click the 'Zenith' brand name and logo at the top of the sidebar."
# description='Navigate to Dashboard via the sidebar menu' process="Click the 'Dashboard' link with the dashboard icon in the sidebar."
# description='Navigate to Customers via the sidebar menu' process="Click the 'Customers' link with the users icon in the sidebar."
# description='Navigate to Orders via the sidebar menu' process="Click the 'Orders' link with the shopping bag icon in the sidebar."
# description='Navigate to Analytics via the sidebar menu' process="Click the 'Analytics' link with the pie‑chart icon in the sidebar."
# description='Open Messages from the SUPPORT section' process="Click the 'Messages' link (showing a badge with number 3) in the SUPPORT section of the sidebar."
# description='Open Settings from the SUPPORT section' process="Click the 'Settings' link in the SUPPORT section of the sidebar."
# description='Log out of the application' process="Click the 'Log Out' button at the bottom of the sidebar."
# description='Search the site' process='Click the search input field in the header, type a query, and press Enter or submit.'
# description='View notifications' process='Click the bell icon button in the header (highlighted with a red dot) to open the notifications panel.'
# description='Open the messaging panel' process='Click the message‑square icon button in the header.'
# description='Select Day view for metrics' process="Click the 'Day' button in the Dashboard Overview header to filter data by day."
# description='Select Week view for metrics' process="Click the 'Week' button in the Dashboard Overview header to filter data by week."
# description='Select Month view for metrics' process="Click the 'Month' button in the Dashboard Overview header to filter data by month."
# description='Open activity details for Devid Heilo' process='Click the activity row showing Devid Heilo, Project Manager, in the Recent Activity panel.'
# description='Open activity details for Henry Fisher' process='Click the activity row showing Henry Fisher, Quality Assurance, in the Recent Activity panel.'
# description='Open activity details for Jhon Doe' process='Click the activity row showing Jhon Doe, Designer, in the Recent Activity panel.'
# description='Open activity details for Wilium Smith' process='Click the activity row showing Wilium Smith, Developer, in the Recent Activity panel.'
# description='Open activity details for Alice Johnson' process='Click the activity row showing Alice Johnson, Data Scientist, in the Recent Activity panel.'

# Information

# description='The application brand displayed in the sidebar header is "Zenith" with a stylized "Z" logo.'
# description='The logged‑in user is John Doe, whose role is Frontend Engineer.'
# description='The main page title reads "Dashboard Overview".'
# description='The overview shows Total Views $3.456K (up 0.43\u202f%), Total Profit $45,2K (up 4.35\u202f%), Total Product 2,450 (up 2.59\u202f%), and Total Users 3.456 (down 0.95\u202f%).'
# description='The chart displays two series: Total Revenue and Total Sales for the period 12.04.2023 – 12.05.2024.'
# description='A "Recent Activity" panel lists recent users’ activities.'