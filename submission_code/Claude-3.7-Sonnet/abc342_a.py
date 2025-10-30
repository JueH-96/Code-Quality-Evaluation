# Read the input string
s = input().strip()

# Find the most common character
most_common = max(set(s), key=s.count)

# Find the position of the character that's different
for i, char in enumerate(s):
    if char != most_common:
        print(i + 1)  # 1-indexed position
        break