S = list(input())
T = list(input())

sl, tl = len(S), len(T)
i = j = 0

ans = []
while i < sl and j < tl:
    if S[i] == T[j]:
        ans.append(str(j + 1))
        i += 1
        j += 1
    else:
        j += 1

print(' '.join(ans))