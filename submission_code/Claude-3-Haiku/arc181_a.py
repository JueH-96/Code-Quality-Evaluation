from typing import List

def solve_permutation(N: int, P: List[int]) -> int:
    operations = 0
    while True:
        # Check if the permutation is already sorted
        if all(P[i] == i+1 for i in range(N)):
            return operations

        # Find the first index where P[i] != i+1
        k = 0
        while k < N and P[k] == k+1:
            k += 1

        # Sort the first k-1 elements in ascending order
        P[:k] = sorted(P[:k])

        # Sort the remaining elements in ascending order
        if k < N:
            P[k:] = sorted(P[k:])

        operations += 1

T = int(input())
for _ in range(T):
    N = int(input())
    P = [int(x) for x in input().split()]
    print(solve_permutation(N, P))