S = input().strip()

def is_extended_abc(s):
    n = len(s)
    
    # Try all possible divisions into A, B, C parts
    for i in range(n+1):  # End of A part
        for j in range(i, n+1):  # End of B part
            # i to j is B part, j to end is C part
            
            # Check if each part contains only allowed letters
            a_part = s[:i]
            b_part = s[i:j]
            c_part = s[j:]
            
            # Check A part - should contain only 'A's
            if not all(c == 'A' for c in a_part):
                continue
                
            # Check B part - should contain only 'B's
            if not all(c == 'B' for c in b_part):
                continue
                
            # Check C part - should contain only 'C's
            if not all(c == 'C' for c in c_part):
                continue
                
            return True
            
    return False

print("Yes" if is_extended_abc(S) else "No")