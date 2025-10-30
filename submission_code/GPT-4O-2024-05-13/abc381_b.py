# YOUR CODE HERE
import sys
input = sys.stdin.read

def is_1122_string(S):
    # Check if the length of S is even
    if len(S) % 2 != 0:
        return "No"
    
    # Check if each character appears exactly twice
    from collections import Counter
    char_count = Counter(S)
    for count in char_count.values():
        if count != 2:
            return "No"
    
    # Check if (2i-1)-th and 2i-th characters are equal
    for i in range(0, len(S), 2):
        if S[i] != S[i + 1]:
            return "No"
    
    return "Yes"

S = input().strip()
print(is_1122_string(S))