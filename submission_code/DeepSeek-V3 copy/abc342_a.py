# YOUR CODE HERE
S = input().strip()
# Determine the majority character
count = {}
for char in S:
    if char in count:
        count[char] += 1
    else:
        count[char] = 1
# Find the character that appears once
for char in count:
    if count[char] == 1:
        target_char = char
        break
# Find the index of the target character (1-based)
for i, char in enumerate(S, 1):
    if char == target_char:
        print(i)
        break