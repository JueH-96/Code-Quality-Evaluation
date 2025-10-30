import re

n = int(input())
s = input()

while True:
    # Replace all occurrences of (lowercase_letters_only)
    new_s = re.sub(r'\([a-z]*\)', '', s)
    if new_s == s:  # No more replacements possible
        break
    s = new_s

print(s)