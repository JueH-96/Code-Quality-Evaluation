S = input()
T = input()

S_idx = 0
T_idx = 0
S_len = len(S)
T_len = len(T)
ans = []

while S_idx < S_len:
    if S[S_idx] == T[T_idx]:
        ans.append(T_idx + 1)
        S_idx += 1
        T_idx += 1
    else:
        T_idx += 1

print(' '.join(map(str, ans)))