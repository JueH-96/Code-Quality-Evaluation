# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
votes = list(map(int, data[2:]))

vote_counts = [0] * (N + 1)
max_votes = 0
winner = 0

for i in range(M):
    candidate = votes[i]
    vote_counts[candidate] += 1
    if vote_counts[candidate] > max_votes:
        max_votes = vote_counts[candidate]
        winner = candidate
    elif vote_counts[candidate] == max_votes:
        winner = min(winner, candidate)
    print(winner)