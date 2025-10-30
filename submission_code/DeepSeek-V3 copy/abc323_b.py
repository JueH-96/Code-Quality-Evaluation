# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:N+1]
    
    wins = [0] * N
    
    for i in range(N):
        for j in range(N):
            if i != j:
                if S[i][j] == 'o':
                    wins[i] += 1
    
    # Create a list of tuples (wins, player number)
    players = [(wins[i], i+1) for i in range(N)]
    
    # Sort first by wins in descending order, then by player number in ascending order
    players.sort(reverse=True, key=lambda x: (x[0], -x[1]))
    
    # Extract the player numbers in the sorted order
    result = [player[1] for player in players]
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()