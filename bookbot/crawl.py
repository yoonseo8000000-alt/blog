import requests
from bs4 import BeautifulSoup
from datetime import datetime
import os

url = "https://www.yes24.com/Product/Category/BestSeller"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

soup = BeautifulSoup(response.text, "html.parser")

titles = soup.select(".gd_name")

blog_path = "/Users/yoonseo/Desktop/blog/content/posts"

for i, title in enumerate(titles[:5], start=1):

    book_title = title.text.strip()

    filename = f"book{i}.md"

    filepath = os.path.join(blog_path, filename)

    content = f"""+++
title = "{book_title}"
date = "{datetime.now().strftime('%Y-%m-%d')}"
draft = false
+++

YES24 베스트셀러 책 추천: {book_title}
"""

    with open(filepath, "w", encoding="utf-8") as file:
        file.write(content)

    print(f"{filename} 생성 완료!")