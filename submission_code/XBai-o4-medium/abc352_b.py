S = input().strip()
T = input().strip()

i = j = 0
res = []
while i < len(S) and j < len(T):
    if S[i] == T[j]:
        res.append(j + 1)
        i += 1
        j += 1
    else:
        j += 1

print(' '.join(map(str, res)))