# YOUR CODE HERE
import sys

def solve():
    """
    This function reads the scores, calculates the runs needed, and prints the result.
    """
    # Read the scores for Team Takahashi across all 9 innings from a single line of input.
    takahashi_runs = list(map(int, sys.stdin.readline().split()))

    # Read the scores for Team Aoki for the first 8 innings from a single line of input.
    aoki_runs = list(map(int, sys.stdin.readline().split()))

    # Calculate the total final score for Team Takahashi by summing their runs.
    takahashi_total_score = sum(takahashi_runs)

    # Calculate the current score for Team Aoki by summing their runs so far.
    aoki_current_score = sum(aoki_runs)

    # To win, Team Aoki's final score must be strictly greater than Team Takahashi's.
    # Let 'x' be the runs Aoki scores in the bottom of the 9th.
    # The winning condition is: aoki_current_score + x > takahashi_total_score
    #
    # Rearranging to solve for x:
    # x > takahashi_total_score - aoki_current_score
    #
    # The difference (takahashi_total_score - aoki_current_score) is the number
    # of runs Aoki needs to tie. To win, they need one more run.
    runs_needed_to_win = (takahashi_total_score - aoki_current_score) + 1

    # Print the minimum number of runs Aoki needs to score to win.
    print(runs_needed_to_win)

# Execute the solution
solve()