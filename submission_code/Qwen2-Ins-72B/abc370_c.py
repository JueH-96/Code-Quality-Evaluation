S = list(input())
T = list(input())
ans = []
while S != T:
    for i in range(len(S)):
        if S[i] != T[i]:
            S[i] = T[i]
            ans.append("".join(S))
            break
print(len(ans))
for i in ans:
    print(i)