import sys
from collections import defaultdict

def solve():
    n, m = map(int, sys.stdin.readline().split())
    votes = list(map(int, sys.stdin.readline().split()))
    count = defaultdict(int)
    max_count = 0
    max_candidates = set()

    for i in range(m):
        count[votes[i]] += 1
        if count[votes[i]] > max_count:
            max_count = count[votes[i]]
            max_candidates = {votes[i]}
        elif count[votes[i]] == max_count:
            max_candidates.add(votes[i])
        print(min(max_candidates))

solve()