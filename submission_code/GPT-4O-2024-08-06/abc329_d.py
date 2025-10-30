# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:]))
    
    # Initialize vote counts for each candidate
    vote_counts = [0] * (N + 1)  # Using 1-based index, so we need N+1
    
    current_winner = 0
    max_votes = 0
    
    results = []
    
    for i in range(M):
        candidate = votes[i]
        vote_counts[candidate] += 1
        
        # Check if this candidate is now the new winner
        if vote_counts[candidate] > max_votes:
            max_votes = vote_counts[candidate]
            current_winner = candidate
        elif vote_counts[candidate] == max_votes:
            # If there's a tie, choose the candidate with the smallest number
            if candidate < current_winner:
                current_winner = candidate
        
        results.append(current_winner)
    
    # Print the results
    for result in results:
        print(result)

if __name__ == "__main__":
    main()