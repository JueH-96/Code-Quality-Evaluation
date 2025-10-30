# YOUR CODE HERE
def is_airport_code(s, t):
    if len(t) != 3 or not t.isupper():
        return False
    
    s = s.upper()
    
    # Check for method 1: subsequence of length 3
    i, j = 0, 0
    while i < len(s) and j < 3:
        if s[i] == t[j]:
            j += 1
        i += 1
    if j == 3:
        return True
    
    # Check for method 2: subsequence of length 2 + 'X'
    if t[2] == 'X':
        i, j = 0, 0
        while i < len(s) and j < 2:
            if s[i] == t[j]:
                j += 1
            i += 1
        if j == 2:
            return True
    
    return False

s = input().strip()
t = input().strip()

print("Yes" if is_airport_code(s, t) else "No")