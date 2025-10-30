# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    votes = list(map(int, data[2:]))
    
    vote_count = [0] * (N + 1)
    current_winner = 0
    max_votes = 0
    
    for i in range(M):
        candidate = votes[i]
        vote_count[candidate] += 1
        
        if vote_count[candidate] > max_votes:
            max_votes = vote_count[candidate]
            current_winner = candidate
        elif vote_count[candidate] == max_votes and candidate < current_winner:
            current_winner = candidate
        
        print(current_winner)

if __name__ == "__main__":
    main()