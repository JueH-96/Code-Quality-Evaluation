import sys
from collections import deque

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    left_max = [0] * N
    right_max = [0] * N

    left_max[0] = A[0]
    for i in range(1, N):
        left_max[i] = max(left_max[i-1], A[i])

    right_max[-1] = A[-1]
    for i in range(N-2, -1, -1):
        right_max[i] = max(right_max[i+1], A[i])

    B = []
    for i in range(N):
        left_part = left_max[i-1] if i > 0 else 0
        right_part = right_max[i+1] if i < N-1 else 0
        max_size = max(left_part, right_part, A[i])
        B.append(max_size)

    print(" ".join(map(str, B)))

if __name__ == "__main__":
    solve()