import sys
from collections import Counter

# Read the input string
s = sys.stdin.readline().strip()

# Count the occurrences of each character
char_counts = Counter(s)

# Find the unique character (the one with count 1)
unique_char = next(char for char, count in char_counts.items() if count == 1)

# Find the position of the unique character (1-based index)
for index, char in enumerate(s):
    if char == unique_char:
        print(index + 1)
        break