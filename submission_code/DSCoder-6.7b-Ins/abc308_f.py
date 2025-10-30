import sys
from typing import List, Tuple

def min_total_cost(N: int, M: int, P: List[int], L: List[int], D: List[int]) -> int:
    # Sort the prices and discounts
    P.sort()
    discounts = sorted(zip(L, D), reverse=True)

    # Apply discounts to the items
    total_cost = 0
    for i in range(N):
        if discounts and P[i] >= discounts[-1][0]:
            total_cost += P[i] - discounts[-1][1]
            discounts.pop()
        else:
            total_cost += P[i]

    return total_cost

def main():
    N, M = map(int, sys.stdin.readline().split())
    P = list(map(int, sys.stdin.readline().split()))
    L = list(map(int, sys.stdin.readline().split()))
    D = list(map(int, sys.stdin.readline().split()))

    print(min_total_cost(N, M, P, L, D))

if __name__ == "__main__":
    main()