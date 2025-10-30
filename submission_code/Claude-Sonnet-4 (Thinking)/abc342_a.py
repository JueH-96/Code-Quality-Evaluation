s = input().strip()

# Count frequency of each character
freq = {}
for char in s:
    freq[char] = freq.get(char, 0) + 1

# Find the position (1-indexed) of the character with frequency 1
for i, char in enumerate(s):
    if freq[char] == 1:
        print(i + 1)
        break