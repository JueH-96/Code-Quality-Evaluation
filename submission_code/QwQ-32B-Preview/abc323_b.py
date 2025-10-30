def main():
    import sys
    # Read the number of players
    N = int(sys.stdin.readline())
    # Read the match results for each player
    S = [sys.stdin.readline().strip() for _ in range(N)]
    
    # Calculate the number of wins for each player
    players = []
    for i in range(N):
        win_count = S[i].count('o')
        player_number = i + 1  # Player numbers start from 1
        players.append( (player_number, win_count) )
    
    # Sort the players: descending win count, ascending player number
    players_sorted = sorted(players, key=lambda x: (-x[1], x[0]))
    
    # Extract the player numbers from the sorted list
    result = [str(p[0]) for p in players_sorted]
    
    # Print the result as space-separated player numbers
    print(' '.join(result))

if __name__ == '__main__':
    main()