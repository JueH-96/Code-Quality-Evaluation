import sys

def solve():
    # Read Team Takahashi's scores for all 9 innings
    # A_scores will be a list of 9 integers
    A_scores_str = sys.stdin.readline().split()
    A_scores = [int(x) for x in A_scores_str]
    
    # Calculate Team Takahashi's total score after 9 innings
    total_score_T = sum(A_scores)
    
    # Read Team Aoki's scores for the first 8 innings
    # B_scores will be a list of 8 integers
    B_scores_str = sys.stdin.readline().split()
    B_scores = [int(x) for x in B_scores_str]
    
    # Calculate Team Aoki's current total score before the bottom of the 9th inning
    current_score_A = sum(B_scores)
    
    # To win, Team Aoki must score strictly more runs than Team Takahashi.
    # Team Takahashi's final score is total_score_T.
    # Therefore, Team Aoki's final score must be at least total_score_T + 1.
    target_score_A_to_win = total_score_T + 1
    
    # Calculate the minimum number of runs Team Aoki needs to score in the bottom of the 9th
    # This is the difference between their target winning score and their current score.
    runs_needed_in_bottom_9th = target_score_A_to_win - current_score_A
    
    # Print the calculated minimum runs
    print(runs_needed_in_bottom_9th)

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()