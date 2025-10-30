def is_one_edit_away(S, T):
    len_s = len(S)
    len_t = len(T)
    
    if abs(len_s - len_t) > 1:
        return False
    
    i = 0
    j = 0
    operations = 0
    
    while i < len_s and j < len_t:
        if S[i] == T[j]:
            i += 1
            j += 1
        else:
            if operations >= 1:
                return False
            if len_s < len_t:
                # Insertion: move j forward
                j += 1
            elif len_s > len_t:
                # Deletion: move i forward
                i += 1
            else:
                # Replacement: move both forward
                i += 1
                j += 1
            operations += 1
    
    # Check for remaining characters in S
    while i < len_s:
        if operations >= 1:
            return False
        i += 1
        operations += 1
    # Check for remaining characters in T
    while j < len_t:
        if operations >= 1:
            return False
        j += 1
        operations += 1
    
    return operations <= 1

# Read inputs
K = int(input())
S = input().strip()
T = input().strip()

# Determine if S can be transformed into T with at most one operation
if is_one_edit_away(S, T):
    print("Yes")
else:
    print("No")