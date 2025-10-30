def can_delete_one(X, Y):
    i = 0  # index for X
    j = 0  # index for Y
    allowed_skips = 1
    while i < len(X) and j < len(Y):
        if X[i] == Y[j]:
            i += 1
            j += 1
        else:
            if allowed_skips == 0:
                return False
            allowed_skips -= 1
            i += 1
    return j == len(Y)

K = int(input())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
else:
    if K == 0:
        print("No")
    else:
        possible = False
        len_S = len(S)
        len_T = len(T)
        if len_S == len_T:
            diff = 0
            for s, t in zip(S, T):
                if s != t:
                    diff += 1
                    if diff > 1:
                        break
            possible = (diff == 1)
        elif len_S + 1 == len_T:
            possible = can_delete_one(T, S)
        elif len_S - 1 == len_T:
            possible = can_delete_one(S, T)
        else:
            possible = False
        print("Yes" if possible else "No")