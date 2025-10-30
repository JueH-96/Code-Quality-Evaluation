S = input().strip()
for i in range(len(S)):
    if S.count(S[i]) == 1:
        print(i + 1)
        break