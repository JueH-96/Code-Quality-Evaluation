# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = data[1:N+1]
    
    # Calculate wins for each player
    wins = [0] * N
    for i in range(N):
        for j in range(N):
            if S[i][j] == 'o':
                wins[i] += 1
    
    # Create a list of (wins, player_number) tuples
    players = [(wins[i], i + 1) for i in range(N)]
    
    # Sort players by number of wins descending, then by player number ascending
    players.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the sorted player numbers
    sorted_player_numbers = [player[1] for player in players]
    
    # Print the result
    print(" ".join(map(str, sorted_player_numbers)))

if __name__ == "__main__":
    main()