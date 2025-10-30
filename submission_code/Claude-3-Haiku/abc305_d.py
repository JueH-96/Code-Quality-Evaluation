import sys

# Read input
N = int(sys.stdin.readline().strip())
A = [int(x) for x in sys.stdin.readline().strip().split()]
Q = int(sys.stdin.readline().strip())
queries = []
for _ in range(Q):
    l, r = [int(x) for x in sys.stdin.readline().strip().split()]
    queries.append((l, r))

# Solve the problem
def solve_query(l, r):
    total_sleep = 0
    for i in range(1, len(A), 2):
        start = A[i-1]
        end = A[i]
        if start >= l and end <= r:
            total_sleep += end - start
        elif start < l and end > r:
            total_sleep += r - l
        elif start < l and end > l:
            total_sleep += end - l
        elif start < r and end > r:
            total_sleep += r - start
    return total_sleep

for l, r in queries:
    print(solve_query(l, r))