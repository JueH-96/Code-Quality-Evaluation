from collections import Counter

s = input()

# Count frequency of each character
char_count = Counter(s)

# Find the character that appears only once and print its position
for char, count in char_count.items():
    if count == 1:
        print(s.index(char) + 1)
        break