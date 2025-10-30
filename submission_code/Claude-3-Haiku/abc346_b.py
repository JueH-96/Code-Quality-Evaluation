# Read the input
W, B = map(int, input().split())

# Define the pattern string
pattern = 'wbwbwwbwbwbw'

# Check if there is a substring of the pattern that matches the given W and B
for i in range(len(pattern)):
    for j in range(i, len(pattern)):
        substring = pattern[i:j+1]
        if substring.count('w') == W and substring.count('b') == B:
            print('Yes')
            return

print('No')