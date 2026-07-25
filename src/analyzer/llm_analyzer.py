import json
import logging
import os
from openai import OpenAI

client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=os.getenv("NVIDIA_API_KEY") or "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"
)

def generate_content_from_article(title, description):
    logging.info(f"Generating content for finance article: {title}")
    prompt = f"""
    Analyze the following Finance/Economic/Business news article:
    Title: {title}
    Description: {description}
    
    Your task:
    1. Understand the main financial, stock market, corporate earnings, or economic policy topic and its significance.
    2. Choose ONE of the following styles that best fits the news:
       - "Market Alert Style"
       - "Economic Insight Style"
       - "Company/Earnings News Style"
       - "Cryptocurrency/Web3 Style"
       - "Global Trade Style"
       - "Personal Finance/Wealth Style"
    3. Generate a hyper-engaging, dramatic, and curiosity-inducing 'headline' (MAX 10 WORDS). This will be written directly on the viral image, so it must grab attention instantly (e.g., "WALL STREET SHOCKED BY NEW FED INTEREST RATE DECREASE!" or "MICROSOFT EARNINGS CRUSH EXPECTATIONS!").
    4. Generate a viral 'hook_text' (1-2 sentences) to complement the headline.
    5. Highlight 1 to 3 important keywords in the headline by wrapping them in asterisks like *this* to make them stand out in a different color.
    6. **STRICT FACEBOOK POLICY CHECK:** Evaluate the article against Facebook Community Standards and Partner Monetization Policies. If the article contains ANY of the following, return a list of flags in the "safety_flags" array:
       - "violence" (physical harm, wars, graphic tragedies)
       - "nudity_sexual" (suggestive content)
       - "hate_speech" (xenophobia, severe insults)
       - "clickbait" (exaggerated or misleading information that withholds key facts)
       - "engagement_bait" (asking users directly to like/comment/share)
       - "tragedy" (self-harm, suicide, mass tragedies)
       - "politics_controversy" (highly polarizing partisan political debates, elections)
       If it violates ANY of these, list the flags (e.g., ["politics_controversy"]). Otherwise, return an empty array [].
    7. Ensure the generated headline and hook_text DO NOT sound like cheap clickbait. Be catchy but factual and professional.
    
    Respond STRICTLY in JSON format with four keys: "headline", "hook_text", "style", and "safety_flags". Do not include markdown formatting or backticks around the JSON.
    """
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=16384,
            extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
            stream=True
        )
        
        response_text = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                response_text += chunk.choices[0].delta.content
                
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        return json.loads(response_text.strip())
    except Exception as e:
        logging.error(f"LLM API failed: {e}")
        return {
            "headline": f"Latest Update: *{title[:50]}*!",
            "hook_text": "A major financial development has surfaced. Here is what we know so far.",
            "style": "Market Alert Style",
            "safety_flags": []
        }

def generate_facebook_caption(title):
    logging.info(f"Generating Facebook caption for: {title}")
    prompt = f"""
    Write a highly engaging Facebook post caption for a finance/economy news page called 'Financial Edge Daily'.
    The post is about this news title: {title}
    
    Requirements:
    - Keep it catchy, professional, informative, and short (3-4 sentences max).
    - Include 2-3 relevant emojis (like 📈, 📉, 💵, 💰, 💼, 🏛️).
    - Include an engaging hook or question at the end to drive comments.
    - Include 5-6 relevant hashtags at the very bottom (like #FinanceNews #StockMarket #Economy #Investing #Wealth #FinancialEdgeDaily).
    - Do not include markdown formatting, just the raw text ready for Facebook.
    """
    try:
        completion = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            top_p=0.95,
            max_tokens=1024,
            stream=False
        )
        caption = completion.choices[0].message.content.strip()
        if not caption:
            raise Exception("Empty response from LLM")
        return caption
    except Exception as e:
        logging.error(f"LLM caption generation failed: {e}")
        return (
            f"📈 Financial Edge Daily Update! 📈\n\n"
            f"{title}\n\n"
            f"What are your thoughts on this latest economic update? Leave a comment below! 👇\n"
            f"#FinanceNews #StockMarket #Economy #Investing #Wealth #FinancialEdgeDaily"
        )
