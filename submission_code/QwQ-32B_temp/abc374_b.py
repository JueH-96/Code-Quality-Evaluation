S = input().strip()
T = input().strip()

if S == T:
    print(0)
else:
    len_S = len(S)
    len_T = len(T)
    max_len = max(len_S, len_T)
    for i in range(1, max_len + 1):
        if i > len_S or i > len_T:
            print(i)
            break
        if S[i-1] != T[i-1]:
            print(i)
            break