import sys
from collections import deque

T = input().strip()
N = int(input().strip())
bags = []
for _ in range(N):
    bag = list(map(str, input().strip().split()))[1:]
    bag.sort(key=len, reverse=True)
    bags.append(bag)

dp = [0] + [-1]*len(T)
q = deque([0])
while q:
    i = q.popleft()
    for bag in bags:
        for s in bag:
            if T[i:i+len(s)] == s and dp[i+len(s)] == -1:
                dp[i+len(s)] = dp[i] + 1
                q.append(i+len(s))

print(dp[-1])