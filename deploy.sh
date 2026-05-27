#!/bin/bash

cd /Users/yoonseo/Desktop/blog

hugo

git add .
git commit -m "auto update"
git push

cd public

git add .
git commit -m "auto deploy"
git push