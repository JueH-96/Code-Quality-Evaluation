import sys
from collections import Counter

def is_1122_string(S):
    # Check if the length of S is even
    if len(S) % 2 != 0:
        return "No"

    # Check if each pair of characters are equal
    for i in range(1, len(S) // 2 + 1):
        if S[2 * i - 2] != S[2 * i - 1]:
            return "No"

    # Check if each character appears exactly zero or two times
    char_count = Counter(S)
    for count in char_count.values():
        if count != 2:
            return "No"

    return "Yes"

# Read input from stdin
S = sys.stdin.read().strip()

# Print the result to stdout
print(is_1122_string(S))