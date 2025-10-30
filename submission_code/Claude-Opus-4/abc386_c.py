# YOUR CODE HERE
K = int(input())
S = input().strip()
T = input().strip()

def can_transform(S, T, K):
    if K != 1:
        return False  # This problem specifically handles K=1
    
    # Case 0: Already equal
    if S == T:
        return True
    
    len_s = len(S)
    len_t = len(T)
    
    # Case 1: Same length - check if exactly one character differs (replace)
    if len_s == len_t:
        diff_count = 0
        for i in range(len_s):
            if S[i] != T[i]:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    # Case 2: S is one character shorter than T (insert)
    elif len_s == len_t - 1:
        i = j = 0
        inserted = False
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i += 1
                j += 1
            elif not inserted:
                inserted = True
                j += 1
            else:
                return False
        return True
    
    # Case 3: S is one character longer than T (delete)
    elif len_s == len_t + 1:
        i = j = 0
        deleted = False
        while i < len_s and j < len_t:
            if S[i] == T[j]:
                i += 1
                j += 1
            elif not deleted:
                deleted = True
                i += 1
            else:
                return False
        return True
    
    # Case 4: Length difference is more than 1
    else:
        return False

if can_transform(S, T, K):
    print("Yes")
else:
    print("No")