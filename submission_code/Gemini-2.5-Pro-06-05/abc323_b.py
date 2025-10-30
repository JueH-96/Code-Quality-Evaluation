# YOUR CODE HERE
import sys

def solve():
    """
    Reads tournament results, ranks players, and prints their numbers.
    """
    try:
        # Read the number of players, N
        N = int(sys.stdin.readline())
    except (ValueError, IndexError):
        # Handle cases with no input
        return

    # A list to store player statistics.
    # Each element will be a tuple: (-wins, player_number).
    # Storing the negative of wins allows us to use the default ascending sort
    # to achieve the desired descending order for win counts.
    players = []

    # Loop through each player, from 1 to N
    for player_number in range(1, N + 1):
        # Read the result string for the current player
        result_string = sys.stdin.readline().strip()
        
        # Count the number of 'o's to determine the number of wins
        wins = result_string.count('o')
        
        # Append the prepared tuple for sorting
        players.append((-wins, player_number))

    # Sort the list of players.
    # Python's default tuple sort is lexicographical. It will:
    # 1. Sort by the first element (-wins) in ascending order. This effectively
    #    sorts by `wins` in descending order.
    # 2. If the first elements are equal (a tie in wins), it sorts by the
    #    second element (player_number) in ascending order.
    # This matches the problem's ranking rules perfectly.
    players.sort()

    # Extract the player numbers from the now-sorted list of tuples.
    ranked_player_numbers = [player[1] for player in players]

    # Print the final list of ranked player numbers, separated by spaces.
    # The `*` operator unpacks the list into positional arguments for print().
    print(*ranked_player_numbers)

# Execute the solution
solve()