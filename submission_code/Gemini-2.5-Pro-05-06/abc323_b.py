# YOUR CODE HERE
def solve():
    N = int(input())
    
    players_data = []
    for i in range(N):
        player_number = i + 1  # Player numbers are 1 to N
        results_string = input()
        wins = results_string.count('o')
        players_data.append({'id': player_number, 'wins': wins})

    # Sort players:
    # 1. Primary key: wins (descending). To achieve descending sort with wins,
    #    we use -wins and sort in ascending order.
    # 2. Secondary key: player_id (ascending).
    # The lambda function returns a tuple (-wins, player_id). Python's default
    # tuple comparison sorts element by element, achieving the desired order.
    sorted_players = sorted(players_data, key=lambda p: (-p['wins'], p['id']))

    # Extract player numbers from the sorted list
    ranked_player_ids = [p['id'] for p in sorted_players]

    # Print the player numbers, space-separated
    print(*ranked_player_ids)

if __name__ == '__main__':
    solve()