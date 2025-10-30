# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input, determines the overall winner based on the game results,
    and prints the winner ('T' for Takahashi, 'A' for Aoki).
    """
    # Read the number of games
    n = int(sys.stdin.readline())
    # Read the game results string
    s = sys.stdin.readline().strip()

    # Count the wins for Takahashi and Aoki
    t_wins = s.count('T')
    # Aoki's wins can be calculated as N - Takahashi's wins
    a_wins = n - t_wins

    # Determine the winner based on the number of wins
    if t_wins > a_wins:
        # Takahashi won more games
        print('T')
    elif a_wins > t_wins:
        # Aoki won more games
        print('A')
    else:
        # The number of wins is equal, apply the tie-breaking rule
        target_wins = t_wins # The score they both reached

        # Track the running count of wins for each player
        t_count = 0
        a_count = 0

        # Iterate through the game results to find who reached the target score first
        for char in s:
            if char == 'T':
                t_count += 1
                # Check if Takahashi reached the target score
                if t_count == target_wins:
                    print('T')
                    return # Winner found, exit the function
            else: # char == 'A'
                a_count += 1
                # Check if Aoki reached the target score
                if a_count == target_wins:
                    print('A')
                    return # Winner found, exit the function

# Call the solve function to execute the logic
solve()