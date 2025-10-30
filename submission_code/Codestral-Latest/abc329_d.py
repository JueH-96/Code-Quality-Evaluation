import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
votes = list(map(int, data[2:]))

# Initialize the vote counts and the current leader
vote_counts = defaultdict(int)
current_leader = None
current_max_votes = 0

# Process each vote
for i in range(M):
    candidate = votes[i]
    vote_counts[candidate] += 1

    # Update the current leader
    if current_leader is None or vote_counts[candidate] > current_max_votes or (vote_counts[candidate] == current_max_votes and candidate < current_leader):
        current_leader = candidate
        current_max_votes = vote_counts[candidate]

    # Print the current leader
    print(current_leader)