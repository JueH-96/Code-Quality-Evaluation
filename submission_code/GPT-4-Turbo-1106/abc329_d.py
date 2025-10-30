from collections import defaultdict
import sys

def main():
    # Read the number of candidates and votes
    N, M = map(int, sys.stdin.readline().split())
    
    # Initialize the votes count for each candidate
    votes_count = defaultdict(int)
    
    # Read the votes and determine the winner after each vote
    for i in range(M):
        vote = int(sys.stdin.readline())
        votes_count[vote] += 1
        
        # Find the current winner
        current_winner, max_votes = 1, votes_count[1]
        for candidate in range(2, N + 1):
            if votes_count[candidate] > max_votes or (votes_count[candidate] == max_votes and candidate < current_winner):
                current_winner, max_votes = candidate, votes_count[candidate]
        
        # Print the current winner
        print(current_winner)

if __name__ == "__main__":
    main()