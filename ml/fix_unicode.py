# -*- coding: utf-8 -*-
import os
import re

def remove_emojis_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove emoji characters
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    content_no_emoji = emoji_pattern.sub('[*]', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content_no_emoji)
    
    print(f"Fixed: {filepath}")

# Fix all Python files in src directory
src_dir = r"d:\Finalyear Projects\Fetal-Abnormality-Detection\ml\src"
for filename in os.listdir(src_dir):
    if filename.endswith('.py'):
        filepath = os.path.join(src_dir, filename)
        remove_emojis_from_file(filepath)

print("All files fixed!")
