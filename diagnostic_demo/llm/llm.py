import os
import litellm
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

system_msg = """\
You are a helpful assistant who is an expert about AI \
regulations as well as fairness and responsible AI.
You always respond in a concise and to-the-point manner."""


def extract_fairness_columns(
    df: pd.DataFrame,
    llm_model: str | None = None,
) -> list[str]:

    litellm.api_key = os.environ.get("LLM_API_KEY", None)
    
    df_str = ""
    for col in df.columns:
        df_str += f"{col}, "
    df_str = df_str[:-2]

    msg = f"""\
Consider the following attributes of a dataset:
```
{df_str}
```

Are there some attributes that could be used by a machine learning model to
discriminate certain groups of people?

Examples of these can be:
age, sex, gender, race, ethnicity, disability status, religion, nationality, family/marital status, 
socioeconomic status, language, sexual orientation, political affiliation, immigration status, ... 

Respond only with a ```python``` code snippet, that contains only a list,
which in turn contains the names of the attributes that may be used for discrimination.
Do not put comments or anything else in the list, since the feature names need to be automatically extracted.
"""

#Respond by, first, citing your reasoning for what could the problematic attributes be.
#After, write in a ```python``` code snippet a list, which contains the attribute names.
#The code snippet must contain the list of attributes and nothing else.

    msgs = [
        {
            "role": "system",
            "content": system_msg,
        },
        {
            "role": "user",
            "content": msg,
        },
    ]

    # openai call
    if llm_model is None:
        llm_model = os.environ.get("LLM_MODEL", "gpt-4o")
    response = litellm.completion(model=llm_model, messages=msgs)

    # extract msg and columns
    msg = response.choices[0].message.content

    # columns follow the python code snippet
    snippet_start = msg.find("```python") + len("```python")
    snippet_end = msg.find("```", snippet_start)

    # remove empty lines
    col_lines = [
        x for x in msg[snippet_start:snippet_end].strip().split("\n") if x != ""
    ]
    col_txt = ""
    for line in col_lines:
        col_txt += line
    # find beginning (first '[') and end (last ']') of the list
    start = col_txt.find("[")
    end = col_txt.rfind("]")
    # extract columns
    columns = col_txt[start + 1: end].split(",")
    columns = [x.replace("'", "").replace('"', "").strip() for x in columns]

    return msg[:snippet_start], columns


def comment_on_fairness_df(
    performance_df: pd.DataFrame,
    llm_model: str = "gpt-4o",
) -> str:
    
    msg = f"""\
Review the following performance metrics of a model, for different groups:
```
{performance_df}
```

Is there significant variability? Is the model fair to different groups?
"""

    msgs = [
        {
            "role": "system",
            "content": system_msg,
        },
        {
            "role": "user",
            "content": msg,
        },
    ]

    # LLM call
    litellm.api_key = os.environ.get("LLM_API_KEY", None)
    response = litellm.completion(model=llm_model, messages=msgs)

    return response.choices[0].message.content