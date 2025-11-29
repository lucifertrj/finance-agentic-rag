from src.client import summarizer_pipeline, gemini_client
from src.config import TAG_GENERATOR_MODEL


def summarize_chunk(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    input_tokens = len(text.split())
    max_length = max(20, min(input_tokens // 2, 128))
    min_length = min(20, max_length - 1)

    try:
        result = summarizer_pipeline(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True
        )
        return result[0]['summary_text']
    except Exception:
        sentences = text.split('. ')
        if len(sentences) > 1:
            return '. '.join(sentences[:2]) + '.'
        return text[:200] if len(text) > 200 else text


def add_doc_type(source_file_name: str) -> str:
    if "Form_8k" in source_file_name or "FORM_8k" in source_file_name:
        return "8-K Filing"
    elif "Form_10K" in source_file_name or "Annural" in source_file_name:
        return "10-K Filing"
    elif "Form_10Q" in source_file_name or "Quarterly" in source_file_name:
        return "10-Q Filing"
    elif "Shareholder" in source_file_name:
        return "Shareholder Letter"
    elif "NETFLIX-BITES" in source_file_name or "Netflix-House" in source_file_name:
        return "News Article"
    else:
        return "Document"


def generate_tags_from_summary(summary: str, chunk_data: str) -> str:
    SYSTEM_PROMPT = f""" You are an expert Financial text analyzer.
    Analyze the provided SUMMARY and ADDITIONAL CONTEXT financial text and \
    identify ALL relevant financial metrics and business concepts mentioned.

    Core Financial Metrics (use these exact tags if relevant):
    - revenue, subscribers, earnings_per_share, operating_income, operating_margin
    - net_income, free_cash_flow, content_spending, debt, cash_and_equivalents
    - advertising_revenue, churn, arpu, guidance

    Be clever and catch:
    - Direct mentions: "revenue grew 15%" → revenue
    - Indirect mentions: "top-line performance" → revenue, "subscriber additions" → subscribers
    - Related concepts: "content investments" → content_spending, "leverage ratios" → debt
    - Business context: "new pricing tiers" → guidance, "ad-supported plan" → advertising_revenue
    - Metrics in context: "margin expansion due to cost controls" → operating_margin

    Respond with relevant tags seperated with comma
    Examples:
    revenue, subscribers,... // this is just example, be smart and generate relevant tags only
    """

    USER_PROMPT = f"""
    Summary to consider: {summary}.
    Additional Context: {chunk_data}
    """

    try:
        response = gemini_client.models.generate_content(
            model=TAG_GENERATOR_MODEL,
            contents=USER_PROMPT,
            config={"system_instruction": SYSTEM_PROMPT}
        )
        return response.text
    except Exception:
        return ""


def update_metadata(data):
    for chunk in data:
        generated_summary = summarize_chunk(chunk.page_content)
        tags = generate_tags_from_summary(generated_summary, chunk.page_content[-800:])
        doc_type = add_doc_type(chunk.metadata['source'])

        chunk.metadata['chunk_summary'] = generated_summary
        chunk.metadata['chunk_tags'] = tags
        chunk.metadata['doc_type'] = doc_type
        chunk.metadata['calendar_year'] = 2025

    return data


DOC_TYPE_MAP = {
    "10-k": "10-K Filing",
    "10k": "10-K Filing",
    "annual": "10-K Filing",
    "10-q": "10-Q Filing",
    "10q": "10-Q Filing",
    "quarterly": "10-Q Filing",
    "8-k": "8-K Filing",
    "8k": "8-K Filing",
    "shareholder": "Shareholder Letter",
    "letter": "Shareholder Letter"
}


def get_target_doc_type(query: str) -> str:
    query_lower = query.lower()
    return next((v for k, v in DOC_TYPE_MAP.items() if k in query_lower), None)

