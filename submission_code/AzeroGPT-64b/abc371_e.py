from collections import defaultdict
import sys

input = sys.stdin.readline

def solve(n, A):
    occurrence = defaultdict(int)
    next_occurrence = defaultdict(lambda: n)
    contribution = 0

    for i in range(n - 1, -1, -1):
        contribution += next_occurrence[A[i]] - i
        occurrence[A[i]] += 1
        next_occurrence[A[i]] = min(next_occurrence[A[i]], i)

    return contribution


n = int(input())
A = list(map(int, input().split()))
print(solve(n, A))