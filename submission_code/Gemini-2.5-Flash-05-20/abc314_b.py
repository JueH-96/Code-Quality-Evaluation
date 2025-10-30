import sys

def solve():
    # Read N, the number of people
    N = int(sys.stdin.readline())

    players_data = []
    for i in range(N):
        # Read C_i, the number of bets for the current person
        C_i = int(sys.stdin.readline())
        
        # Read A_i, the outcomes the current person bet on
        # Convert to a set for efficient lookup (checking if X is in A_i)
        A_i = set(map(int, sys.stdin.readline().split()))
        
        # Store player's information: original 1-based ID, bet count, and their set of bets
        players_data.append({
            'id': i + 1,  # Player IDs are 1-based
            'bets_count': C_i,
            'bets': A_i
        })

    # Read X, the outcome of the roulette spin
    X = int(sys.stdin.readline())

    eligible_players = []
    min_bets_found = float('inf') # Initialize with a very large number

    # First pass: Identify all players who bet on X and find the minimum number of bets among them.
    for player in players_data:
        if X in player['bets']:
            # This player bet on X, so they are eligible
            eligible_players.append((player['bets_count'], player['id']))
            
            # Update the minimum bets found so far among eligible players
            min_bets_found = min(min_bets_found, player['bets_count'])

    result_players = []

    # Second pass: Filter eligible players to include only those who made the minimum number of bets.
    # This loop will only run if eligible_players is not empty (i.e., someone bet on X).
    if eligible_players:
        for bets_count, player_id in eligible_players:
            if bets_count == min_bets_found:
                result_players.append(player_id)
    
    # Sort the final list of player IDs in ascending order as required.
    result_players.sort()

    # Output the count of players
    sys.stdout.write(str(len(result_players)) + '
')

    # Output the player IDs, space-separated.
    # If result_players is empty, ' '.join(map(str, [])) results in '',
    # and sys.stdout.write('
') ensures a newline, which matches sample output for K=0.
    sys.stdout.write(' '.join(map(str, result_players)) + '
')

# Call the solve function to run the program
solve()