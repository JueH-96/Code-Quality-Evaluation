N, D = map(int, input().split())
S = input()
count = 0
for i in range(N):
    if S[i] == '@':
        D -= 1
        if D == 0:
            break
print(N - D)