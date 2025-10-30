import sys

# Read N, the number of people.
try:
    n = int(sys.stdin.readline())
except (ValueError, IndexError):
    # This handles edge cases like empty input.
    # Given the problem constraints, N will be >= 1, but this is robust.
    n = 0

# Read the bet information for each person.
# We will store this in a list of dictionaries, where each dictionary
# represents a player and contains their ID, number of bets, and the bets themselves.
all_players_data = []
for i in range(1, n + 1):
    # C_i: the number of bets for person i
    c = int(sys.stdin.readline())
    # A_i: the numbers person i has bet on. Using a set for efficient lookups.
    bets = set(map(int, sys.stdin.readline().split()))
    all_players_data.append({'id': i, 'count': c, 'bets': bets})

# Read X, the winning number.
# If n=0, we won't read X, but the program will correctly output 0.
if n > 0:
    x = int(sys.stdin.readline())
else:
    x = -1 # A dummy value, won't be used.

# Find all players who bet on the winning number X.
winners = []
for player in all_players_data:
    if x in player['bets']:
        winners.append(player)

# If no one won, print the specified output and exit.
if not winners:
    print(0)
    print()
else:
    # Among the winners, find the minimum number of bets.
    min_bet_count = min(player['count'] for player in winners)
    
    # Collect the IDs of all winners who have this minimum number of bets.
    final_winner_ids = []
    for player in winners:
        if player['count'] == min_bet_count:
            final_winner_ids.append(player['id'])
    
    # The list of IDs is already sorted because we processed players in order (1 to N).
    
    # Print the final count and the IDs.
    print(len(final_winner_ids))
    print(*final_winner_ids)