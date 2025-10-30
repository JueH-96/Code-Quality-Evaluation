import sys

def solve():
    """
    Solves the Takahashi vs Aoki game problem.
    """
    # Read the number of games, N.
    # While N is given, it's implicitly defined by the length of the result string S.
    try:
        n = int(sys.stdin.readline())
        # Read the string of game results, S.
        s = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle potential empty input or invalid format
        return

    # Count the total number of wins for Takahashi ('T').
    takahashi_wins = s.count('T')
    # The number of wins for Aoki ('A') is the total games minus Takahashi's wins.
    aoki_wins = n - takahashi_wins

    # Condition 1: Check if one player has more wins than the other.
    if takahashi_wins > aoki_wins:
        # If Takahashi has more wins, Takahashi is the winner.
        print('T')
    elif aoki_wins > takahashi_wins:
        # If Aoki has more wins, Aoki is the winner.
        print('A')
    else:
        # Condition 2 (Tie-breaker): Both players have the same number of wins.
        # The winner is the one who reached this number of wins first.
        # As reasoned above, the player who won the last game was "catching up",
        # meaning the other player had already achieved the final score.
        
        if s[-1] == 'T':
            # If Takahashi won the last game, Aoki reached the score first.
            print('A')
        else:  # s[-1] == 'A'
            # If Aoki won the last game, Takahashi reached the score first.
            print('T')

solve()