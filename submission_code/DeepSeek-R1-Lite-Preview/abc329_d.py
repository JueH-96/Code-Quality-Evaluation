import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    votes = list(map(int, input[2:2+M]))
    
    vote_counts = [0] * (N + 1)
    current_max = 0
    current_winner = 0
    winners = []
    
    for vote in votes:
        vote_counts[vote] += 1
        if vote_counts[vote] > current_max:
            current_max = vote_counts[vote]
            current_winner = vote
        elif vote_counts[vote] == current_max:
            if vote < current_winner:
                current_winner = vote
        winners.append(str(current_winner))
    
    print('
'.join(winners))

if __name__ == "__main__":
    main()