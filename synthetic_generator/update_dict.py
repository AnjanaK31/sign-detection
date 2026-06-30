import os
import string

dict_path = r"d:\Cogentic\sign-detection\synthetic_generator\dict.txt"

# Original symbols from teammate
symbols = [
    '±', 'Ø', '°', '>', '<', '=', '"', 'µ', '-', '+', '×', '*', '⊕', '☒', '☐', '⌖'
]

# Additional standard characters
digits = list(string.digits)
uppercase = list(string.ascii_uppercase)
lowercase = list(string.ascii_lowercase)
punctuation = [' ', '.', ',', ':', '/', '(', ')', '&']

# Combine all unique characters
all_chars = []
for char_list in [symbols, digits, uppercase, lowercase, punctuation]:
    for c in char_list:
        if c not in all_chars:
            all_chars.append(c)

with open(dict_path, 'w', encoding='utf-8') as f:
    for c in all_chars:
        f.write(c + '\n')

print(f"Updated {dict_path} with {len(all_chars)} characters.")
