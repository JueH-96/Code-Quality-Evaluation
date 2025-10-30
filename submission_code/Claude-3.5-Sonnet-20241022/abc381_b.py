# Read input string
S = input().strip()

def is_1122_string(s):
    # Check condition 1: length must be even
    if len(s) % 2 != 0:
        return False
    
    # Check condition 2: adjacent pairs must be equal
    for i in range(0, len(s), 2):
        if s[i] != s[i+1]:
            return False
    
    # Check condition 3: each character must appear exactly 0 or 2 times
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    
    for count in char_count.values():
        if count != 2:
            return False
    
    return True

# Print result
if is_1122_string(S):
    print("Yes")
else:
    print("No")