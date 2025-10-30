# YOUR CODE HERE
S = input().strip()
T = input().strip()

def is_airport_code(S, T):
    # Case 1: T is a subsequence of S of length 3
    if len(T) == 3:
        i = 0
        for char in S:
            if i < 3 and char.upper() == T[i]:
                i += 1
        if i == 3:
            return True
    # Case 2: T is a subsequence of S of length 2 plus 'X'
    if T[2] == 'X':
        i = 0
        for char in S:
            if i < 2 and char.upper() == T[i]:
                i += 1
        if i == 2:
            return True
    return False

if is_airport_code(S, T):
    print("Yes")
else:
    print("No")