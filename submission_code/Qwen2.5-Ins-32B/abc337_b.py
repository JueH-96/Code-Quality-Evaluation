def is_extended_abc_string(s):
    if not s:
        return "Yes"
    
    a_end = -1
    b_end = -1
    
    for i, char in enumerate(s):
        if char != 'A' and a_end == -1:
            a_end = i - 1
        if char != 'B' and b_end == -1 and a_end != -1:
            b_end = i - 1
    
    if a_end == -1 or b_end == -1:
        return "No"
    
    for i in range(a_end + 1):
        if s[i] != 'A':
            return "No"
    
    for i in range(a_end + 1, b_end + 1):
        if s[i] != 'B':
            return "No"
    
    for i in range(b_end + 1, len(s)):
        if s[i] != 'C':
            return "No"
    
    return "Yes"

s = input().strip()
print(is_extended_abc_string(s))