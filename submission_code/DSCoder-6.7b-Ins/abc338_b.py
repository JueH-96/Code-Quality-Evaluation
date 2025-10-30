from collections import Counter
import sys

S = sys.stdin.readline().strip()

# Count the frequency of each character in S
counter = Counter(S)

# Find the maximum frequency
max_freq = max(counter.values())

# Find the characters with the maximum frequency
max_freq_chars = [k for k, v in counter.items() if v == max_freq]

# The earliest character in alphabetical order is the first character in the list
earliest_char = sorted(max_freq_chars)[0]

print(earliest_char)