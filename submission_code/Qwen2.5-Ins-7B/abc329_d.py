# YOUR CODE HERE
from collections import defaultdict

N, M = map(int, input().split())
votes = list(map(int, input().split()))
count = defaultdict(int)
max_count = 0
winner = 0

for i, vote in enumerate(votes):
    count[vote] += 1
    if count[vote] > max_count or (count[vote] == max_count and vote < winner):
        max_count = count[vote]
        winner = vote
    print(winner)