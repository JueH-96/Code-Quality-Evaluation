# YOUR CODE HERE
from itertools import combinations

def solve():
    N, M = map(int, input().split())
    stands = [input().strip() for _ in range(N)]

    for k in range(1, N + 1):
        for combo in combinations(range(N), k):
            flavors = set()
            for stand in combo:
                flavors.update(i for i, flavor in enumerate(stands[stand]) if flavor == 'o')
            if len(flavors) == M:
                return k

    return N  # This line should never be reached given the constraints

print(solve())