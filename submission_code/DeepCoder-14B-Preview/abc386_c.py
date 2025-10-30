def can_insert(a, b):
    if len(b) != len(a) + 1:
        return False
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
            if j - i > 1:
                return False
    return True

K = int(input())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
else:
    if len(S) == len(T):
        diff = sum(1 for s, t in zip(S, T) if s != t)
        if diff == 1:
            print("Yes")
        else:
            print("No")
    else:
        if abs(len(S) - len(T)) != 1:
            print("No")
        else:
            if len(S) > len(T):
                if can_insert(T, S):
                    print("Yes")
                else:
                    print("No")
            else:
                if can_insert(S, T):
                    print("Yes")
                else:
                    print("No")