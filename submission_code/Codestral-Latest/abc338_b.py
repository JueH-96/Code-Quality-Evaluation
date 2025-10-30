from collections import Counter

# Read the input string
S = input().strip()

# Count the frequency of each character
char_count = Counter(S)

# Find the maximum frequency
max_freq = max(char_count.values())

# Find the character(s) with the maximum frequency
max_freq_chars = [char for char, count in char_count.items() if count == max_freq]

# Find the earliest character in alphabetical order
earliest_char = min(max_freq_chars)

# Print the result
print(earliest_char)