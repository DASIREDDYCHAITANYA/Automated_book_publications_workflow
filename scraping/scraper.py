import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import os

async def scrape_chapter_async(url, save_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)

        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        content_div = soup.find("div", id="mw-content-text")

        text = content_div.get_text(separator="\n")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text)

        await browser.close()
        print(f"✅ Chapter saved to: {save_path}")
url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
save_path = "data/raw/chapter_1.txt"
await scrape_chapter_async(url, save_path)
