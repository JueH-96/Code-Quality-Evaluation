from collections import Counter

# Read input
N, M = map(int, input().split())
votes = list(map(int, input().split()))

# Initialize vote counts
vote_counts = [0] * (N + 1)

# Process each vote
for i in range(M):
    # Update vote counts
    vote_counts[votes[i]] += 1
    
    # Find the winner
    winner = min(range(1, N+1), key=lambda x: (-vote_counts[x], x))
    
    # Print the winner
    print(winner)