import sys

# Read input
N = int(input())
scores = [int(x) for x in input().split()]

# Determine ranks
ranks = [0] * N
current_rank = 1
while any(rank == 0 for rank in ranks):
    max_score = max(score for i, score in enumerate(scores) if ranks[i] == 0)
    num_tied = sum(1 for score in scores if score == max_score and ranks[scores.index(score)] == 0)
    for i, score in enumerate(scores):
        if score == max_score and ranks[i] == 0:
            ranks[i] = current_rank
    current_rank += num_tied

# Print output
for rank in ranks:
    print(rank)