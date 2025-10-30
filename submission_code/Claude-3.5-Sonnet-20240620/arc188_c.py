# YOUR CODE HERE
from itertools import product

def check_consistency(N, testimonies, confused):
    for honest in product([0, 1], repeat=N):
        valid = True
        for a, b, c in testimonies:
            a_honest = honest[a-1] ^ confused[a-1]
            b_honest = honest[b-1]
            if a_honest:
                if (b_honest and c == 1) or (not b_honest and c == 0):
                    valid = False
                    break
            else:
                if (b_honest and c == 0) or (not b_honest and c == 1):
                    valid = False
                    break
        if valid:
            return True
    return False

def solve(N, M, testimonies):
    for confused in product([0, 1], repeat=N):
        if check_consistency(N, testimonies, confused):
            return ''.join(map(str, confused))
    return '-1'

# Read input
N, M = map(int, input().split())
testimonies = [tuple(map(int, input().split())) for _ in range(M)]

# Solve and print output
print(solve(N, M, testimonies))