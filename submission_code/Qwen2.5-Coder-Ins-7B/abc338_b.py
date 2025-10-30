# YOUR CODE HERE
import sys
from collections import Counter

# Read input from stdin
S = sys.stdin.read().strip()

# Count the frequency of each character
char_count = Counter(S)

# Find the maximum frequency
max_freq = max(char_count.values())

# Find the characters with the maximum frequency
max_chars = [char for char, freq in char_count.items() if freq == max_freq]

# Sort the characters alphabetically and print the first one
print(sorted(max_chars)[0])