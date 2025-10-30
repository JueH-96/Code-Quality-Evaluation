def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    # Calculate wins for each player
    wins = []
    for i in range(N):
        s = input_data[i+1]
        wins.append(s.count('o'))
    
    # Create list of players (player number is 1-indexed)
    players = list(range(1, N + 1))
    
    # Sort players first by descending wins, then by ascending player number if wins are equal
    players.sort(key=lambda player: (-wins[player-1], player))
    
    # Print the sorted players
    print(" ".join(map(str, players)))

if __name__ == '__main__':
    main()