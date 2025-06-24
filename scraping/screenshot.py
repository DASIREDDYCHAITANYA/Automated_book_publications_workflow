import asyncio
from playwright.async_api import async_playwright
import os

async def take_screenshot_async(url, save_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        await page.screenshot(path=save_path, full_page=True)
        await browser.close()
        print(f"✅ Screenshot saved to: {save_path}")
url = "https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1"
screenshot_path = "data/screenshots/chapter_1.png"

await take_screenshot_async(url, screenshot_path)