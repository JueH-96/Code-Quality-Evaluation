from itertools import product

def soln():
    N, K = map(int, input().split())
    R = list(map(int, input().split()))

    sequences = []
    for seq in product(*[range(1, r+1) for r in R]):
        if sum(seq) % K == 0:
            sequences.append(seq)

    sequences.sort()
    for seq in sequences:
        print(" ".join(map(str, seq)))

soln()