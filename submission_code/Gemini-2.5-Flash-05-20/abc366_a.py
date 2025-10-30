# YOUR CODE HERE
import sys

def solve():
    # Read N, T, A from standard input
    N, T, A = map(int, sys.stdin.readline().split())

    # Calculate the number of remaining votes
    remaining_votes = N - (T + A)

    # Check if Takahashi is guaranteed to win
    # This happens if Takahashi's current votes are strictly greater than
    # Aoki's maximum possible votes (current A + all remaining votes)
    takahashi_guaranteed_win = (T > A + remaining_votes)

    # Check if Aoki is guaranteed to win
    # This happens if Aoki's current votes are strictly greater than
    # Takahashi's maximum possible votes (current T + all remaining votes)
    aoki_guaranteed_win = (A > T + remaining_votes)

    # If either candidate is guaranteed to win, the outcome is decided
    if takahashi_guaranteed_win or aoki_guaranteed_win:
        print("Yes")
    else:
        print("No")

# Call the solve function to execute the program
solve()