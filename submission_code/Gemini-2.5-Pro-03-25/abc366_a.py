# YOUR CODE HERE
import sys

def main():
    # Read input N, T, A from standard input
    line = sys.stdin.readline().split()
    n = int(line[0]) # Total number of votes (odd)
    t = int(line[1]) # Current votes for Takahashi
    a = int(line[2]) # Current votes for Aoki

    # Calculate the minimum number of votes required to guarantee a win.
    # Since N is odd, a candidate needs strictly more than N/2 votes to win.
    # The smallest integer number of votes greater than N/2 is floor(N/2) + 1.
    # In Python, integer division `//` computes the floor.
    min_votes_to_win = (n // 2) + 1

    # Check if Takahashi has already secured enough votes to guarantee a win.
    # This happens if Takahashi's current vote count T is already >= min_votes_to_win.
    # If T >= min_votes_to_win, even if all remaining votes go to Aoki,
    # Takahashi's final count will still be T, which is enough to win.
    takahashi_wins_guaranteed = (t >= min_votes_to_win)

    # Check if Aoki has already secured enough votes to guarantee a win.
    # This happens if Aoki's current vote count A is already >= min_votes_to_win.
    # If A >= min_votes_to_win, even if all remaining votes go to Takahashi,
    # Aoki's final count will still be A, which is enough to win.
    aoki_wins_guaranteed = (a >= min_votes_to_win)

    # The outcome of the election is decided if either candidate is guaranteed to win.
    if takahashi_wins_guaranteed or aoki_wins_guaranteed:
        # If either Takahashi or Aoki has enough votes to guarantee a win,
        # regardless of how the remaining votes are cast, the outcome is decided.
        print("Yes")
    else:
        # If neither candidate has secured enough votes yet, the outcome is still undecided.
        # The remaining votes could potentially change the winner.
        print("No")

if __name__ == '__main__':
    # Run the main function when the script is executed
    main()
# END OF YOUR CODE HERE