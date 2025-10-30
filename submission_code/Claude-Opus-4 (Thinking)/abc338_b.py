# YOUR CODE HERE
s = input()

# Count frequency of each character
freq = {}
for char in s:
    freq[char] = freq.get(char, 0) + 1

# Find the maximum frequency
max_freq = max(freq.values())

# Find and print the earliest alphabetical character with maximum frequency
for char in sorted(freq.keys()):
    if freq[char] == max_freq:
        print(char)
        break