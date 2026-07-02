#!/bin/bash
cd /Users/yoonseo/bookbot
python3 crawl.py
cd /Users/yoonseo/Desktop/blog
hugo --cleanDestinationDir
git submodule update --init --recursive
git add .
git commit -m "auto update"
git push
cd public
git add .
git commit -m "auto deploy"
git push
