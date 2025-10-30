# YOUR CODE HERE
import sys

def main():
    N = int(input())
    results = [input().strip() for _ in range(N)]
    
    # Calculate the number of wins for each player
    wins = [0] * N
    for i in range(N):
        for j in range(N):
            if results[i][j] == 'o':
                wins[i] += 1
    
    # Create a list of (wins, -player_number) to sort by wins and then by player number
    ranked_players = [(wins[i], -i) for i in range(N)]
    
    # Sort the players by wins (descending) and player number (ascending)
    ranked_players.sort(reverse=True)
    
    # Extract the player numbers and print them in descending order of rank
    ranked_player_numbers = [-player[1] for player in ranked_players]
    print(" ".join(map(str, ranked_player_numbers)))

if __name__ == "__main__":
    main()