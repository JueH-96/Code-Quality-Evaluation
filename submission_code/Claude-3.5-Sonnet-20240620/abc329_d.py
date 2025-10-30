# YOUR CODE HERE
N, M = map(int, input().split())
votes = list(map(int, input().split()))

vote_count = [0] * (N + 1)  # +1 to make indexing easier
current_winner = 0
max_votes = 0

for i, vote in enumerate(votes, 1):
    vote_count[vote] += 1
    
    if vote_count[vote] > max_votes:
        max_votes = vote_count[vote]
        current_winner = vote
    elif vote_count[vote] == max_votes and vote < current_winner:
        current_winner = vote
    
    print(current_winner)