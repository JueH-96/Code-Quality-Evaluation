K = int(input())
S = input()
T = input()

def can_transform(S, T):
    if S == T:
        return True
    
    len_S = len(S)
    len_T = len(T)
    
    # Case: Replace operation
    if len_S == len_T:
        diff_count = 0
        for i in range(len_S):
            if S[i] != T[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    # Case: Delete operation (S is longer by 1)
    elif len_S == len_T + 1:
        i, j = 0, 0
        skipped = False
        
        while i < len_S and j < len_T:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if skipped:
                    return False
                skipped = True
                i += 1  # Skip this character in S
        
        return j == len_T
    
    # Case: Insert operation (T is longer by 1)
    elif len_T == len_S + 1:
        i, j = 0, 0
        skipped = False
        
        while i < len_S and j < len_T:
            if S[i] == T[j]:
                i += 1
                j += 1
            else:
                if skipped:
                    return False
                skipped = True
                j += 1  # Skip this character in T
        
        return i == len_S
    
    # All other cases
    else:
        return False

if can_transform(S, T):
    print("Yes")
else:
    print("No")