from collections import Counter

# Read the input string
S = input().strip()

# Count the frequency of each character in S
freq_counter = Counter(S)

# Find the maximum frequency
max_freq = max(freq_counter.values())

# Find the character that appears most frequently and comes earliest in alphabetical order
most_frequent_char = min([char for char in freq_counter if freq_counter[char] == max_freq])

# Print the result
print(most_frequent_char)