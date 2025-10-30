# YOUR CODE HERE
S = input().strip()
T = input().strip()

def is_airport_code(S, T):
    if len(T) != 3:
        return False
    # Case 1: T is a subsequence of S of length 3
    i = 0
    j = 0
    while i < len(S) and j < 3:
        if S[i].upper() == T[j]:
            j += 1
        i += 1
    if j == 3:
        return True
    # Case 2: T is a subsequence of S of length 2 plus 'X'
    if T[2] == 'X':
        i = 0
        j = 0
        while i < len(S) and j < 2:
            if S[i].upper() == T[j]:
                j += 1
            i += 1
        if j == 2:
            return True
    return False

if is_airport_code(S, T):
    print("Yes")
else:
    print("No")