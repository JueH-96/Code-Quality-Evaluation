# YOUR CODE HERE
S = input().strip()

# Count frequency of each character
char_count = {}
for char in S:
    char_count[char] = char_count.get(char, 0) + 1

# Find the character that appears only once
unique_char = None
for char, count in char_count.items():
    if count == 1:
        unique_char = char
        break

# Find the position of the unique character (1-indexed)
for i in range(len(S)):
    if S[i] == unique_char:
        print(i + 1)
        break