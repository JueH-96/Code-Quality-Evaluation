def determine_winners():
    # Read input
    N, M = map(int, input().split())
    votes = list(map(int, input().split()))
    
    # Initialize vote count array and winner tracking
    vote_count = [0] * (N+1)  # Use 1-based indexing to match candidate numbers
    current_winner = 0
    max_votes = 0
    
    # Process each vote and determine the winner after each
    for i in range(M):
        candidate = votes[i]
        vote_count[candidate] += 1
        
        # Update the winner if:
        # 1. This candidate now has more votes than anyone else
        # 2. This candidate ties for most votes but has a lower number
        if vote_count[candidate] > max_votes:
            max_votes = vote_count[candidate]
            current_winner = candidate
        elif vote_count[candidate] == max_votes and candidate < current_winner:
            current_winner = candidate
        
        # Output the current winner
        print(current_winner)

if __name__ == "__main__":
    determine_winners()