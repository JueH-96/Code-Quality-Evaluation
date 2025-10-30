# YOUR CODE HERE
def is_1122_string(s):
    # Check if length is even
    if len(s) % 2 != 0:
        return False
    
    # Check if adjacent pairs are equal and count occurrences
    char_count = {}
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            return False
        char_count[s[i]] = char_count.get(s[i], 0) + 2
    
    # Check if each character appears exactly twice
    if any(count != 2 for count in char_count.values()):
        return False
    
    return True

# Read input
S = input().strip()

# Check if S is a 1122 string and print result
if is_1122_string(S):
    print("Yes")
else:
    print("No")