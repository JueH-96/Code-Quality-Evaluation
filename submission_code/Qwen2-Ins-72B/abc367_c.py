from itertools import product

N, K = map(int, input().split())
R = list(map(int, input().split()))

ans = []
for seq in product(*[range(1, r+1) for r in R]):
    if sum(seq) % K == 0:
        ans.append(seq)

for seq in sorted(ans):
    print(*seq)