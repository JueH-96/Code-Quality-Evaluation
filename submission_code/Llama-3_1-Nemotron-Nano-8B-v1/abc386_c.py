def is_subsequence(a, b):
    i = 0
    j = 0
    len_a = len(a)
    len_b = len(b)
    while i < len_a and j < len_b:
        if a[i] == b[j]:
            i += 1
        j += 1
    return i == len_a

K = int(input())
S = input().strip()
T = input().strip()

if S == T:
    print("Yes")
else:
    lenS, lenT = len(S), len(T)
    if abs(lenS - lenT) > 1:
        print("No")
    else:
        if lenS == lenT:
            diff = sum(c1 != c2 for c1, c2 in zip(S, T))
            print("Yes" if diff == 1 else "No")
        elif lenS < lenT:
            if lenT != lenS + 1:
                print("No")
            else:
                print("Yes" if is_subsequence(S, T) else "No")
        else:
            if lenS != lenT + 1:
                print("No")
            else:
                print("Yes" if is_subsequence(T, S) else "No")