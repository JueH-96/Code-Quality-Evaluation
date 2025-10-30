# YOUR CODE HERE

S = input()

# Count the frequency of each character
freq = {}
for char in S:
    if char in freq:
        freq[char] += 1
    else:
        freq[char] = 1

# Find the character with the maximum frequency
max_freq = max(freq.values())
max_chars = [char for char, freq in freq.items() if freq == max_freq]

# Sort the characters in alphabetical order and print the first one
max_chars.sort()
print(max_chars[0])