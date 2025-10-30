# YOUR CODE HERE
def solve():
    # Read scores for Team Takahashi
    # These are space-separated integers on a single line
    a_scores_input = input().split()
    # Convert string scores to integers
    a_scores = [int(s) for s in a_scores_input]

    # Read scores for Team Aoki (up to 8th inning)
    # These are also space-separated integers on a single line
    b_scores_input = input().split()
    # Convert string scores to integers
    b_scores = [int(s) for s in b_scores_input]

    # Calculate total score for Team Takahashi after 9 innings
    score_takahashi = sum(a_scores)

    # Calculate total score for Team Aoki after 8 innings
    score_aoki_8th = sum(b_scores)

    # Let X be the runs Aoki scores in the bottom of the 9th.
    # Aoki's final score will be score_aoki_8th + X.
    # Takahashi's final score is score_takahashi.
    # For Aoki to win, their final score must be strictly greater than Takahashi's score:
    # score_aoki_8th + X > score_takahashi
    #
    # Rearranging for X:
    # X > score_takahashi - score_aoki_8th
    #
    # Since X must be an integer, the minimum value for X is:
    # (score_takahashi - score_aoki_8th) + 1
    #
    # The problem states score_takahashi >= score_aoki_8th.
    # Let diff = score_takahashi - score_aoki_8th. So, diff >= 0.
    # The runs needed is diff + 1, which will be >= 1.

    runs_needed_to_win = (score_takahashi - score_aoki_8th) + 1

    # Print the result
    print(runs_needed_to_win)

if __name__ == '__main__':
    solve()