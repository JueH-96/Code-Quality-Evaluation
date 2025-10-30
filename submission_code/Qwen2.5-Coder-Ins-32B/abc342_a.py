# YOUR CODE HERE
import sys

def find_unique_char_index(s):
    if s[0] != s[1]:
        return 1 if s[0] == s[2] else 0
    for i in range(2, len(s)):
        if s[i] != s[0]:
            return i

input = sys.stdin.read().strip()
result = find_unique_char_index(input)
print(result + 1)  # Convert 0-based index to 1-based index