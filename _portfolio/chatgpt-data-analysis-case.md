---
title: "What Do People Really Ask ChatGPT?"
excerpt: "A Data Analysis of Prompt Structure and Response Readability<br/><img src='/images/0chatgpt-data-analysis/chatgpt-data-0.png'>"
collection: portfolio
---


Ever wonder if ChatGPT talks like a textbook or a friend? I got curious and decided to find out — so I pulled a dataset of ~52,000 real ChatGPT instruction-response pairs and ran some analysis on it.

Here's what I found.

![Editing a Markdown file for a talk](/images/0chatgpt-data-analysis/chatgpt-data-0.png)



## The Data

I used the [alpaca-gpt4 dataset](https://huggingface.co/datasets/vicgalle/alpaca-gpt4) from Hugging Face, which contains real prompts paired with GPT-4 generated responses. It loads in one line:

```python
df = pd.read_parquet("hf://datasets/vicgalle/alpaca-gpt4/data/train-00000-of-00001-6ef3991c06080e14.parquet")
```

Three columns matter here: `instruction` (the prompt), `input` (optional extra context), `output` (ChatGPT's response) and `text` (the full training sentence that combines the instruction, input, and output).




## 1. What Do People Actually Ask ChatGPT?

First question: what topics dominate? I built a word cloud from all the prompts after stripping out generic instruction verbs like *write*, *create*, *list* — otherwise those would drown everything out.

```python
custom_stopwords = [
    'write', 'generate', 'create', 'give', 'list', 'describe', 'explain',
    'provide', 'make', 'find', 'suggest', 'classify', 'rewrite', 'summarize',
    # ... and more prompt boilerplate
]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

all_text = " ".join(df["instruction"].apply(clean_text))
wordcloud = WordCloud(width=1200, height=800, stopwords=stop_words).generate(all_text)
```

The result? Topics like *story*, *email*, *recipe*, *sentence*, *product*, and *code* dominate. People are mostly using ChatGPT as a writing assistant and task executor — not a search engine.

![Word Cloud](/images/0chatgpt-data-analysis/wordcloud.png)


## 2. How Are Prompts Structured?

I wrote a simple rule-based classifier to bucket prompts by their opening phrase:

```python
def categorize_prompt(text):
    text_lower = text.lower()
    if text_lower.startswith(('write', 'create', 'generate', 'compose', 'draft')):
        return 'Creative Task'
    elif text_lower.startswith(('explain', 'describe', 'define', 'clarify')):
        return 'Explanation'
    elif text_lower.startswith(('give', 'list', 'provide', 'name', 'outline')):
        return 'Listing Task'
    elif text_lower.startswith(('can you', 'could you', 'would you')):
        return 'Question'
    # ... more categories
```

**Creative Tasks** came out on top by a wide margin, followed by **Listing Tasks** and **Explanations**. People lean heavily toward asking ChatGPT to *make* things rather than just *explain* them.
<div id="chart1"></div>





## 3. Are ChatGPT Answers Easy to Read?

This is where it gets interesting. I used the [Flesch Reading Ease score](https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests) — a standard formula that considers sentence length and syllable count. Higher score = easier to read (think children's book). Lower score = harder (think academic journal).

```python
import textstat

df["flesch_score"] = df["output"].apply(textstat.flesch_reading_ease)

def readability_level(score):
    if score >= 90:   return "Very Easy"
    elif score >= 60: return "Easy"
    elif score >= 30: return "Medium"
    elif score >= 10: return "Difficult"
    else:             return "Very Difficult"
```

The majority of responses scored in the **"Easy" to "Medium"** range — roughly the level of a magazine article or popular blog post. ChatGPT doesn't write like a PhD thesis, but it's not dumbing things down either. It sits comfortably in the middle.
<div id="chart2"></div>




## 4. Do Longer Prompts Get More Complex Answers?

Here's a hypothesis worth testing: if you write a longer, more detailed prompt, does ChatGPT respond with more complex language?

```python
df["instruction_word_count"] = df["instruction"].apply(lambda x: len(x.split()))

fig = px.scatter(
    df,
    x="instruction_word_count",
    y="flesch_score",
    trendline="ols",
    title="Prompt Length vs Readability Scores",
    opacity=0.5
)
```

The regression line shows a **slight negative trend** — meaning longer prompts are weakly associated with *lower* (harder) readability scores. But the effect is small and there's a ton of variance. The real takeaway: prompt length alone isn't a strong predictor of answer complexity.
<div id="chart3"></div>


## 5. How Verbose Is ChatGPT?

I split each response into sentences and calculated words-per-sentence:

```python
def sentence_count(text):
    sentences = re.split(r'[.!?]', str(text))
    return len([s for s in sentences if s.strip()])

df["words_per_sentences"] = df["output_word_count"] / df["sentence_count"]
```

The box plot told a clean story: **Q1 to Q3 falls between ~10 and ~21 words per sentence**. That's solidly in "moderate" territory — not terse bullet points, not run-on academic prose. ChatGPT writes like someone who took a good writing class.

<div id="chart4"></div>



## 6. Does Adding Context Actually Help?

The dataset has an optional `input` field where users can add extra context to their prompt. I wanted to know: does including context change the response in any meaningful way?

```python
df["has_input"] = df["input"].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
)

comparison = df.groupby("has_input")[["output_word_count", "flesch_score"]].mean()
```

Results: prompts **with** extra context got **longer responses** on average, but the **readability scores were similar**. So context makes ChatGPT more verbose, but not necessarily harder or easier to read. Interesting — it adapts quantity, not complexity.

<div id="chart5"></div>



## Takeaways

Here's the short version of what I learned:

- **People mostly use ChatGPT to create things** — write emails, generate stories, draft content.
- **Responses land in "easy-to-medium" readability** — roughly magazine-level, not textbook-level.
- **Prompt length has little effect on answer complexity** — how you frame the question matters more than how long it is.
- **ChatGPT writes in moderate-length sentences** — typically 10–21 words, which is natural and readable.
- **Extra context leads to longer (but not harder) answers** — if you want more detail, give more context.



## Tools Used

- `pandas` — data wrangling
- `nltk` + `wordcloud` — text preprocessing and visualization
- `textstat` — readability scoring
- `plotly` + `seaborn` — interactive and static charts
- `numpy` — regression calculations

So, does ChatGPT talk like a textbook or a friend? The answer is… a bit of both! Next time you ask a question, consider how you phrase it — the way you write your prompt can nudge the AI toward clarity, creativity, or casual conversation.

<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>
renderPlot("chart1", "/assets/charts/0chatgpt-analysis/prompt_types.json");
renderPlot("chart2", "/assets/charts/0chatgpt-analysis/readability_levels.json");
renderPlot("chart3", "/assets/charts/0chatgpt-analysis/prompt_vs_readability.json");
renderPlot("chart4", "/assets/charts/0chatgpt-analysis/sentence_length_box.json");
renderPlot("chart5", "/assets/charts/0chatgpt-analysis/context_effect.json");
</script>