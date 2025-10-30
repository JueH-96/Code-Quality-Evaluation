import sys
from collections import defaultdict
from heapq import *

N, M = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

votes = defaultdict(int)
candidates = []

for i in range(M):
    votes[A[i]] += 1
    while True:
        candidate, vote = candidates[0]
        if votes[candidate] < vote:
            heappop(candidates)
        else:
            break
    heappush(candidates, (-votes[A[i]], A[i]))
    print(candidates[0][1])