# AI Lease Insights – Rent Roll Summary & Validation Assistant

This project is designed to help analysts and real estate professionals extract or generate summaries from rent roll documents, which can often be messy, inconsistent, or incomplete. The goal is to automate what’s normally a manual and time-consuming task—gathering high-level metrics from these files—by using a combination of programmatic logic and large language models (LLMs). 

The tool supports Excel, CSV, and PDF documents and allows both batch processing and an interactive web interface to review results, generate insights, and compare properties. It’s built to be modular, easy to use, and adaptable to new formats or requirements down the line.

**Live App**: [AI Lease Insights](https://arvikj-aileaseinsights.streamlit.app/)

---

## Why I Built This

I wanted to solve a real problem that shows up often in property operations: taking large, unstructured, or inconsistently formatted rent roll files and converting them into a clean, useful summary. I started with simple parsing techniques, then experimented with rule-based systems, and finally shifted to using LLMs once I saw how flexible and efficient they could be for this kind of semi-structured data.

Along the way, I focused not only on getting results, but on building something that would be understandable, extendable, and genuinely useful to others. I tried to make design choices that reflected thoughtfulness and responsibility—building with clarity, handling errors gracefully, and aiming for real impact rather than just technical completeness.

---

## Project Features

- Works with **Excel (.xlsx, .xls), CSV, and PDF** rent roll formats.
- Automatically detects if a document includes a summary section, and either extracts or generates a structured JSON summary.
- Uses the **Gemini 2.5 Flash Preview (04-17)** model, which was released less than a week ago, and supports **adjustable thinking budgets** for more accurate summarization.
- Integrates Google's new `google-genai` client (required for budget control), replacing the older SDK that will lose support after August 2025.
- Employs a hybrid retry strategy—using higher thinking budgets first, and falling back to lower ones if the call takes too long or fails.
- Interactive **Streamlit web interface** lets you upload files, view summaries, get AI-generated insights, and compare properties visually.
- Uses **three different Gemini models** internally, each one selected for a specific task (summary extraction, quick single-file insight, and multi-file comparisons). This improves speed and avoids hitting API rate limits during high usage.
- Includes **error handling, input validation, and type coercion** for safety and reliability.

---

## A Quick Example

Here’s an example of the structured output the tool generates(More examples of actual outputs can be found in the `Outputs` directory):

```json
{
  "summary_section_found": true,
  "total_units": 16,
  "total_actual_rent": 14527.0,
  "average_rent": 907.94,
  "occupancy_rate": 0.9375,
  "total_market_rent": 16075.0,
  "total_square_feet": 13630.0,
  "status_breakdown": [
    {
      "status": "Occupied No Notice",
      "count": 15
    },
    {
      "status": "Vacant Unrented Not Ready",
      "count": 1
    }
  ],
  "charge_codes": [
    {
      "charge_type": "garage_rent",
      "total_amount": 500.0
    },
    {
      "charge_type": "pet_rent_monthly",
      "total_amount": 150.0
    },
    {
      "charge_type": "rent",
      "total_amount": 14527.0
    }
  ]
}
```
---

## My Approach

### Where I Started

I initially tried building a rule-based system using a mix of libraries: `pandas` for structured spreadsheets, `pdfplumber` for PDFs, `openpyxl` for Excel cell access, and even `fuzzywuzzy` for string matching. The idea was to locate headers, find keywords like "summary," and then parse values from specific cells or ranges.

This worked on a few files, but the more I tested it, the less sustainable it felt. The rent roll documents had too much variation—multi-level headers, merged cells, multiple summary tables in different parts of the file, inconsistent naming conventions, and numeric columns formatted as strings. In some cases, even the unit-level data was split across multiple rows or interrupted by blank lines. These issues made deterministic parsing very brittle and unpredictable.

### Why I Moved to LLMs

LLMs seemed like a natural next step. They’re better at handling messiness, context-switching, varied phrasing, or missing values. Instead of creating dozens of parsing rules, I could just feed the model a cleaned version of the file and ask it for the exact structure I wanted.

To make that possible, I wrote a custom file parser that extracts the raw text and table content from Excel, PDF, and CSV files in a consistent format. That parsed data—along with a small metadata block (row counts, page counts, etc.)—gets passed to the LLM as part of the prompt. This reduces the complexity the model has to reason through and makes its outputs more consistent.

I chose to use the **gemini-2.5-flash-preview-04-17** model. Released very recently (04-17-2025), this model supports a `thinking_budget` parameter, which lets you control how much the model "thinks" before responding, which is especially useful for longer files. For small files, I set the budget to 0 for speed. For larger ones, I scale it up—but cap it to avoid runaway thinking time.

That said, I did run into some strange behavior. Sometimes, even for smaller files, Gemini would get stuck in what felt like a "thinking loop"—it would exceed 4 minutes and then return no output, just a JSON parse error. Despite the budget being limited, the model would sometimes exceed the practical limits. To fix that, I implemented a timeout and retry strategy: if the model times out or fails once, I retry the same prompt but with no thinking budget the second time. This drastically improved stability.

Another interesting thing I noticed was that even though most rent roll files are small in size—just a few kilobytes—the number of tokens can be huge. Some files went beyond 70,000 tokens after parsing. That made my choice of using Gemini even more justified. The model has a large million token context window, performs well at long-text reasoning, and has a very generous free-tier quota compared to other options.

### Why Gemini (and Not Other Models)

I considered other LLM APIs and open-source alternatives, but there were trade-offs:

- **Open-source models like Mistral, LLaMA, or Mixtral**: These were appealing but weren’t practical to run locally due to hardware constraints. Fine-tuning and then locally running these models would require time, higher compute on cloud platforms for hosting the app, and more data than I had available.
- **Other free APIs** (e.g., Cohere, HuggingFace Inference endpoints): These lacked the performance or flexibility I needed for JSON extraction and had tighter rate limits.

I also considered using the newer **Gemini 2.5 Pro Experimental**, but it felt like overkill. It’s optimized for deeper, more creative reasoning or conversational tasks. For structured summary generation and extraction, Flash performed faster and well within expectations.

So I ended up using three Gemini models for different tasks in the Streamlit app:

- **Summary generation**: 2.5 Flash with adjustable thinking budget.
- **Quick insights**: 2.0 Flash Lite for faster turnarounds.
- **Multi-file comparisons**: A separate 2.0 Flash call to avoid hitting rate limits and specialize the reasoning and handle multiple files.

This setup helps balance performance and quota management in particular as each model has its own quota. It also keeps the app fast and responsive under different usage patterns and overall makes the system more resilient.


---

## Streamlit Web App Overview

Once I had the backend working, I wanted to make it easier to explore the results. Streamlit made sense—it’s fast to build, doesn’t require any infrastructure, and is free to host for smaller apps.

Here’s what you can do with the app:

- Upload files (PDF, Excel, or CSV) or a ZIP with multiple documents.
- For each file, view:
  - A detected or generated summary in JSON format.
  - A "quick insight" section using a lighter model for fast feedback.
- Compare files visually—charts for occupancy, rent distribution, charge codes, etc.
- Use a separate comparison insight generator that analyzes all selected files and provides AI-generated takeaways.

These extra features weren’t necessary per say, but I added them because I think they make the tool more useful and approachable. They're useful for providing better clarity and usability and personally make the system feel a lot more impactful for potential stakeholders.

---

## How to Run This

### 1. Clone the repo

```bash
git clone https://github.com/your-username/rent-roll-analyzer.git
cd rent-roll-analyzer
```

### 2. Set up a virtual environment (using venv or conda)

If you're using Anaconda or Miniconda, you can create and activate a new environment like this:

```bash
conda create --name leaseinsights python=3.11 -y
conda activate leaseinsights
```

or if you prefer to use venv:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a file called `.env` in the project root:

```env
GOOGLE_API_KEY="your-gemini-api-key"
```

### 5. Run the app locally

```bash
streamlit run app.py
```

---

## File Structure

```
src/
├── file_parser.py        # Handles document parsing
├── data_processing.py    # Metadata, serialization, validation
├── llm_interface.py      # Gemini API interactions
├── pipeline.py           # File-level orchestration
├── main.py               # CLI batch runner
Outputs/                  # Summary outputs as JSON
sample_data/              # Input test documents
.env                      # Stores your API key
app.py                    # Streamlit app entry point
```

---

## Known Limitations

- This system operates within the limits of Gemini's free-tier API quotas. For summary generation, there are 500 calls per day, with a cap of 15 calls and 250,000 tokens per minute. For insights, the daily limit increases to 1500 calls. Hitting those limits can cause retries and delays.
- Depending on the size and structure of the file, the LLM may take longer or shorter to respond. While retry logic is in place to handle rate limits or timeouts, there could be occasional failures if the model gets stuck or the file is especially complex.
- Very complex or unconventional document layouts may still confuse the LLM.
- The model fails to be accurate for some particular weirdly structured large input data with lots of empty rows in between different unit IDs in a column for example.
- The outputs are extremely close and accurate to the ground truth based upon manual validation but there are minor deviations from actual calculated values at times, showcasing the lack of the model's ability to calculate reliably.
- The JSON output may include `null` values if the model can’t confidently extract a field.
- I rely on the model to follow the prompt instructions, but like all LLMs, it can occasionally misinterpret something or hallucinate a value even though I have set the temperature to 0 for now.
- No strict validation against ground truth summaries. Future versions could include a comparison mode with rule-based sanity checks if the data can be cleaned better.

---

## Ideas for the Future

There are a few directions I’d like to explore further if time and resources allow:

- **Fine-tuning or prompt tuning**: With more labeled examples, I could fine-tune a smaller model or apply even better prompt-engineering strategies to improve consistency and reduce hallucinations—especially for documents that follow a predictable structure.

- **Running open-source models locally**: Tools like Ollama, LM Studio, or lightweight models from HuggingFace (e.g., Mistral or Phi-2) could be evaluated for offline use. This would require more compute and a larger and cleaner dataset for testing, but it could open up more control over inference and response times.

- **Configurable validation rules**: Future versions could include a module that performs arithmetic recalculations from unit-level data, checking the LLM-generated summary against deterministic results where possible.

- **Caching and rate-limit resilience**: Saving results locally or in a lightweight database (e.g., SQLite) could help prevent duplicate processing and make the app faster over time in case of mutliple calls for the same file.

- **Backend infrastructure and analytics expansion**: Moving the application to AWS with a structured Flask backend would open the door to better error handling, persistent storage, user access control, and more scalable usage. This infrastructure would also make it easier to integrate with larger rent-roll data pipelines—allowing the summaries to feed into reporting dashboards, analytics layers, or compliance tools used by asset managers. It would also enable richer visualizations and insights that go beyond the single-session interaction model of Streamlit.

These are all possible next steps that would build on what’s already working and make the tool more useful in production settings.


## Final Thoughts

This project came out of trying to solve a messy real-world problem. Rent rolls aren’t standardized, and getting clean, consistent summaries from them is harder than it looks. I tried multiple strategies before settling on a mix of programmatic logic and LLM-based reasoning, and I’ve learned a lot through the process.

I approached this with a mindset of curiosity, persistence, and practicality. Wherever possible, I added fallbacks, made logic configurable, and built tools that others could adapt or extend. The Streamlit interface isn’t just a wrapper—it’s a way to test and learn from the model outputs in a way that feels natural.

Thanks for checking this out.
