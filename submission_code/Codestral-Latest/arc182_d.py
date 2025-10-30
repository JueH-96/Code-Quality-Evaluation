import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:]))

def min_operations_to_match(A, B, M):
    if A == B:
        return 0

    diff = [abs(a - b) for a, b in zip(A, B)]
    max_diff = max(diff)

    if max_diff > M // 2:
        diff = [M - d if d > M // 2 else d for d in diff]

    return sum(diff)

result = min_operations_to_match(A, B, M)
print(result)