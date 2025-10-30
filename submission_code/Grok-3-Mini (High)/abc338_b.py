from collections import Counter

# Read the input string from stdin
s = input().strip()

# Count the frequency of each character
counter = Counter(s)

# Find the maximum frequency
max_freq = max(counter.values())

# Find the character with max frequency that is smallest alphabetically
result = min(char for char in counter if counter[char] == max_freq)

# Print the result
print(result)