import sys
import bisect

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

# Process each day
for i in range(1, N + 1):
    # Find the first firework day on or after the i-th day
    index = bisect.bisect_left(A, i)
    if index < M:
        print(A[index] - i)
    else:
        print(0)