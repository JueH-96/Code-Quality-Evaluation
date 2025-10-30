N = int(input())
S = [input() for _ in range(N)]

rev_S = [s[::-1] for s in S]

for i in range(N):
    for j in range(N):
        if i != j and S[i] in rev_S:
            print('Yes')
            exit()

print('No')