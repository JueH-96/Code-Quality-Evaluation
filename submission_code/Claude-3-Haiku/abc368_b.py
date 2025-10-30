from typing import List

def solve(N: int, A: List[int]) -> int:
    count = 0
    while len([x for x in A if x > 0]) > 1:
        A.sort(reverse=True)
        A[0] -= 1
        A[1] -= 1
        count += 1
    return count

# Read input
N = int(input())
A = list(map(int, input().split()))

# Solve and print the answer
print(solve(N, A))