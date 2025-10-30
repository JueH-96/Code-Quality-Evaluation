# YOUR CODE HERE

from itertools import product

def solve(N, K, R):
    sequences = []
    for seq in product(*(range(1, r+1) for r in R)):
        if sum(seq) % K == 0:
            sequences.append(seq)
    sequences.sort()
    return sequences

N, K = map(int, input().split())
R = list(map(int, input().split()))

sequences = solve(N, K, R)
for seq in sequences:
    print(*seq)