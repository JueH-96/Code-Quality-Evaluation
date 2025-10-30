# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N, M = int(data[0]), int(data[1])
    A = list(map(int, data[2:]))
    
    votes = [0] * (N + 1)
    winners = []
    
    for i in range(M):
        votes[A[i]] += 1
        max_votes = max(votes)
        for j in range(1, N + 1):
            if votes[j] == max_votes:
                winners.append(j)
                break
    
    for winner in winners:
        print(winner)

if __name__ == "__main__":
    main()