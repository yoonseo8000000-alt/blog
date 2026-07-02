#!/bin/bash
cd /Users/yoonseo/bookbot
python3 crawl.py
cd /Users/yoonseo/Desktop/blog
find public -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
hugo
git add .
git commit -m "auto update"
git push
cd public
git add .
git commit -m "auto deploy"
git push
