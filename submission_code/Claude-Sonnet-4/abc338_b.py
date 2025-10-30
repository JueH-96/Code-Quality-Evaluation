# YOUR CODE HERE
S = input().strip()

# Count frequency of each character
freq = {}
for char in S:
    freq[char] = freq.get(char, 0) + 1

# Find maximum frequency
max_freq = max(freq.values())

# Find all characters with maximum frequency
max_chars = [char for char, count in freq.items() if count == max_freq]

# Return the earliest alphabetically
result = min(max_chars)
print(result)