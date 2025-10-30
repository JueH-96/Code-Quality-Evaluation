import sys
from bisect import bisect_right
from itertools import accumulate
from collections import defaultdict

def read_input():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    return N, A

def solve(N, A):
    sorted_A = sorted((a, i) for i, a in enumerate(A))
    prefix_sum = list(accumulate(A))
    prefix_count = list(accumulate(1 for _ in A))
    count = defaultdict(int)
    result = [0]*N
    total = 0
    for a, i in sorted_A:
        pos = bisect_right(sorted_A, (a, N))
        if pos < N:
            count[sorted_A[pos][0]] -= 1
        total += a * (N - pos)
        result[i] = total - prefix_sum[i] + count[A[i]] * A[i]
    return result

def main():
    N, A = read_input()
    result = solve(N, A)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()