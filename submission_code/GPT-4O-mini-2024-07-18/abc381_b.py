def is_1122_string(S):
    # Check if the length of S is even
    if len(S) % 2 != 0:
        return "No"
    
    # Check if each character appears exactly twice
    from collections import Counter
    char_count = Counter(S)
    
    if any(count != 2 for count in char_count.values()):
        return "No"
    
    # Check if the string follows the pairing condition
    for i in range(0, len(S), 2):
        if S[i] != S[i + 1]:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Output the result
print(is_1122_string(S))