N, M = map(int, input().split())
votes = list(map(int, input().split()))

# Initialize vote counts for each candidate
counts = [0] * (N + 1)  # index 0 is unused, candidates are 1-indexed

for i in range(M):
    # Count the i-th vote
    candidate = votes[i]
    counts[candidate] += 1
    
    # Find the current winner
    max_votes = 0
    winner = 1
    
    for j in range(1, N + 1):
        if counts[j] > max_votes:
            max_votes = counts[j]
            winner = j
    
    print(winner)