from itertools import product

def solve():
    N, K = map(int, input().split())
    R = list(map(int, input().split()))
    for seq in product(*[range(1, r+1) for r in R]):
        if sum(seq) % K == 0:
            print(*seq)

solve()