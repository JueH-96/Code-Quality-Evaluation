# YOUR CODE HERE
S = input().strip()

# Count frequency of each character
freq = {}
for char in S:
    freq[char] = freq.get(char, 0) + 1

# Find the maximum frequency
max_freq = max(freq.values())

# Find all characters with maximum frequency
max_chars = []
for char, count in freq.items():
    if count == max_freq:
        max_chars.append(char)

# Sort alphabetically and print the first one
max_chars.sort()
print(max_chars[0])