import sys

# Read input
N = int(sys.stdin.readline().strip())
contests = []
for _ in range(N):
    L, R = map(int, sys.stdin.readline().strip().split())
    contests.append((L, R))
Q = int(sys.stdin.readline().strip())
queries = [int(sys.stdin.readline().strip()) for _ in range(Q)]

# Solve the problem
def solve(initial_rating):
    rating = initial_rating
    for L, R in contests:
        if L <= rating <= R:
            rating += 1
    return rating

for query in queries:
    print(solve(query))