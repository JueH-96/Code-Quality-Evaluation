# YOUR CODE HERE
import sys
from collections import defaultdict

input = sys.stdin.read().split()
N = int(input[0])
M = int(input[1])
votes = list(map(int, input[2:]))

vote_count = defaultdict(int)
current_winner = 1
max_votes = 0

for vote in votes:
    vote_count[vote] += 1
    if vote_count[vote] > max_votes or (vote_count[vote] == max_votes and vote < current_winner):
        current_winner = vote
        max_votes = vote_count[vote]
    print(current_winner)