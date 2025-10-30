import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:]))
    
    # Initialize vote counts for each candidate
    vote_counts = [0] * (N + 1)
    
    # This will store the current winner and the maximum votes
    current_winner = 0
    max_votes = 0
    
    results = []
    
    for i in range(M):
        candidate = votes[i]
        vote_counts[candidate] += 1
        
        # Check if the current vote changes the winner
        if vote_counts[candidate] > max_votes or (vote_counts[candidate] == max_votes and candidate < current_winner):
            current_winner = candidate
            max_votes = vote_counts[candidate]
        
        results.append(str(current_winner))
    
    # Print all results
    print("
".join(results))

if __name__ == "__main__":
    main()