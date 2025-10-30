def is_1122_string(s):
    if len(s) % 2 != 0:
        return "No"
    
    # Check if every pair of characters are the same
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            return "No"
    
    # Check if every character appears exactly twice
    char_count = {}
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    for count in char_count.values():
        if count != 2:
            return "No"
    
    return "Yes"

# Read input
import sys
input = sys.stdin.read
s = input().strip()

# Output result
print(is_1122_string(s))