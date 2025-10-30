# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+M]))
    
    # Initialize vote counts
    vote_counts = [0] * (N + 1)
    current_winner = 0
    max_votes = 0
    
    for i in range(M):
        candidate = A[i]
        vote_counts[candidate] += 1
        if vote_counts[candidate] > max_votes:
            max_votes = vote_counts[candidate]
            current_winner = candidate
        elif vote_counts[candidate] == max_votes:
            if candidate < current_winner:
                current_winner = candidate
        print(current_winner)

if __name__ == "__main__":
    main()