from httpcore import __name
from transformers import AutoTokenizer

from src.extractor import bmodel


def extract_features(file):
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
    content = file.read()

    # tokens = tokenizer.encode(content)

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
    return features

if __name__ == '__main__':
    with open("src/dummy-data/dashboard.html") as file:
        features = extract_features(file)

        if features:
            print('\nAction\n')
            for action in features.actions:
                print(action)
            print('\nInformation\n')
            for info in features.info_:
                print(info)
        else:
            print('Model did not perform tool call. Please Retry.')
    
# Created By Amit Mahapatra