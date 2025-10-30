S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    for i in range(1, max(len(S), len(T)) + 1):
        if i > len(S) or i > len(T):
            # One string is shorter - difference found
            print(i)
            break
        elif S[i-1] != T[i-1]:
            # Characters at position i are different
            print(i)
            break