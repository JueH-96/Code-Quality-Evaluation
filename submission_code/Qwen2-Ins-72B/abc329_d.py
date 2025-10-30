N, M = map(int, input().split())
votes = list(map(int, input().split()))

# Initialize the vote counts and the current winner
vote_counts = [0] * (N + 1)
current_winner = 0
max_votes = 0

# Process each vote and print the current winner
for i, vote in enumerate(votes):
    vote_counts[vote] += 1
    if vote_counts[vote] >= max_votes:
        max_votes = vote_counts[vote]
        current_winner = vote
    print(current_winner)