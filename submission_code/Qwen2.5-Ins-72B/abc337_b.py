# YOUR CODE HERE
def is_extended_abc_string(s):
    a_count = 0
    b_count = 0
    c_count = 0
    
    for char in s:
        if char == 'A':
            if b_count > 0 or c_count > 0:
                return False
            a_count += 1
        elif char == 'B':
            if c_count > 0:
                return False
            b_count += 1
        elif char == 'C':
            c_count += 1
    
    return a_count >= 0 and b_count >= 0 and c_count >= 0

s = input().strip()
if is_extended_abc_string(s):
    print("Yes")
else:
    print("No")