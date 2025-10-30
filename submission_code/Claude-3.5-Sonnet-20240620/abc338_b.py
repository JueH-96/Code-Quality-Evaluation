# YOUR CODE HERE
from collections import Counter

# Read input string
S = input().strip()

# Count character frequencies
char_counts = Counter(S)

# Find the maximum frequency
max_freq = max(char_counts.values())

# Get all characters with the maximum frequency
most_frequent = [char for char, count in char_counts.items() if count == max_freq]

# Sort the most frequent characters alphabetically
most_frequent.sort()

# Print the first (earliest in alphabetical order) character
print(most_frequent[0])