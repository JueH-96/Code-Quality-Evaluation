# YOUR CODE HERE
def is_extended_abc(s):
    a_end = 0
    while a_end < len(s) and s[a_end] == 'A':
        a_end += 1
    
    b_start = a_end
    b_end = b_start
    while b_end < len(s) and s[b_end] == 'B':
        b_end += 1
    
    c_start = b_end
    return all(ch == 'C' for ch in s[c_start:])

s = input().strip()
print("Yes" if is_extended_abc(s) else "No")