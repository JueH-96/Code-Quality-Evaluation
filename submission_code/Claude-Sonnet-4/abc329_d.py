# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Initialize vote counts for each candidate (1-indexed)
votes = [0] * (N + 1)

# Process each vote
for i in range(M):
    # Count the current vote
    candidate = A[i]
    votes[candidate] += 1
    
    # Find the winner after this vote
    max_votes = 0
    winner = 1
    
    for j in range(1, N + 1):
        if votes[j] > max_votes:
            max_votes = votes[j]
            winner = j
        elif votes[j] == max_votes and j < winner:
            winner = j
    
    print(winner)