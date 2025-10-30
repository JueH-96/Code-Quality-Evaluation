def is_subsequence(s, t):
    """Check if t is a subsequence of s."""
    j = 0
    for i in range(len(s)):
        if j < len(t) and s[i] == t[j]:
            j += 1
    return j == len(t)

def is_airport_code(s, t):
    # Convert S to uppercase for easier comparison
    s_upper = s.upper()
    
    # Case 1: Check if T is formed by a subsequence of length 3 from S
    if is_subsequence(s_upper, t):
        return True
    
    # Case 2: Check if T is formed by a subsequence of length 2 from S with 'X' appended
    if t[-1] == 'X' and is_subsequence(s_upper, t[:-1]):
        return True
    
    return False

# Read input
s = input().strip()
t = input().strip()

# Determine if T is an airport code for S
if is_airport_code(s, t):
    print("Yes")
else:
    print("No")