---
title: "What Do People Really Ask ChatGPT?"
excerpt: "A Data Analysis of Prompt Structure and Response Readability<br/><img src='/images/0chatgpt-data-analysis/chatgpt-data-0.png'>"
collection: portfolio
---

![Editing a Markdown file for a talk](/images/0chatgpt-data-analysis/chatgpt-data-0.png)

Ever wonder if ChatGPT talks like a textbook or a friend? I got curious and decided to find out — so I pulled a dataset of ~52,000 real ChatGPT instruction-response pairs and ran some analysis on it.

Here's what I found.



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

<div id="chart"></div>

<script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
<script>
fetch("assets/charts/prompt_types.json")
.then(r => r.json())
.then(fig => Plotly.newPlot("chart", fig.data, fig.layout, {responsive:true}));
</script>

```plotly
{"data":[{"hovertemplate":"prompt_type=%{x}\u003cbr\u003ecount=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"","orientation":"v","showlegend":false,"textposition":"auto","x":["Creative Task","Listing Task","Explanation","Editing\u002fRewriting","Classification","Advice","Problem Solving","Question"],"xaxis":"x","y":{"dtype":"i2","bdata":"ZS6gHMoTvAdtBWIEEANzAA=="},"yaxis":"y","type":"bar"}],"layout":{"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"prompt_type"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"count"}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of ChatGPT Prompt Types"},"barmode":"relative"}}
```



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

<iframe src="charts/readability_levels.html" width="100%" height="500"></iframe>

```plotly

```



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

<iframe src="charts/prompt_vs_readability.html" width="100%" height="500"></iframe>

```plotly

```

## 5. How Verbose Is ChatGPT?

I split each response into sentences and calculated words-per-sentence:

```python
def sentence_count(text):
    sentences = re.split(r'[.!?]', str(text))
    return len([s for s in sentences if s.strip()])

df["words_per_sentences"] = df["output_word_count"] / df["sentence_count"]
```

The box plot told a clean story: **Q1 to Q3 falls between ~10 and ~21 words per sentence**. That's solidly in "moderate" territory — not terse bullet points, not run-on academic prose. ChatGPT writes like someone who took a good writing class.

<iframe src="charts/sentence_length_box.html" width="100%" height="500"></iframe>

```plotly

```

## 6. Does Adding Context Actually Help?

The dataset has an optional `input` field where users can add extra context to their prompt. I wanted to know: does including context change the response in any meaningful way?

```python
df["has_input"] = df["input"].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
)

comparison = df.groupby("has_input")[["output_word_count", "flesch_score"]].mean()
```

Results: prompts **with** extra context got **longer responses** on average, but the **readability scores were similar**. So context makes ChatGPT more verbose, but not necessarily harder or easier to read. Interesting — it adapts quantity, not complexity.

<iframe src="charts/context_effect.html" width="100%" height="500"></iframe>

```plotly

```

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