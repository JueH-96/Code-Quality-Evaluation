# YOUR CODE HERE
N, M = map(int, input().split())
votes = list(map(int, input().split()))

# Initialize vote counts for each candidate
vote_count = [0] * (N + 1)  # Index 0 unused, candidates are 1 to N

# Process each vote
for i in range(M):
    # Add the current vote
    candidate = votes[i]
    vote_count[candidate] += 1
    
    # Find the winner (candidate with most votes, ties broken by smallest number)
    max_votes = 0
    winner = 1
    
    for j in range(1, N + 1):
        if vote_count[j] > max_votes:
            max_votes = vote_count[j]
            winner = j
    
    print(winner)