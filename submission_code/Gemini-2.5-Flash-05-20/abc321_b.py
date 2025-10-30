import sys

def solve():
    # Read N (total number of rounds) and X (target final grade)
    N, X = map(int, sys.stdin.readline().split())
    
    # Read the scores from the first N-1 rounds
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize the answer to -1. This will be printed if no score in round N
    # can achieve the target grade X.
    ans = -1

    # Iterate through all possible integer scores for round N, from 0 to 100.
    # We are looking for the minimum score, so we check in increasing order.
    for score_N in range(0, 101):
        # Create a temporary list containing all N scores:
        # the N-1 given scores and the current candidate score_N for the last round.
        current_scores = A + [score_N]
        
        # Sort the list of N scores in ascending order.
        # This is necessary to identify the lowest, highest, and the middle scores.
        current_scores.sort()

        # Calculate the final grade according to the problem's rule:
        # Sum of the N-2 scores excluding the lowest and the highest.
        # In a sorted list (S_1, S_2, ..., S_N), this means summing S_2 through S_{N-1}.
        # In Python, for a list `current_scores`, this corresponds to `current_scores[1]`
        # up to `current_scores[N-2]`.
        # The slice `current_scores[1:N-1]` extracts elements from index 1 up to (N-1)-1 = N-2.
        current_grade = sum(current_scores[1:N-1])

        # Check if the calculated grade meets or exceeds the target grade X.
        if current_grade >= X:
            # If it does, we have found the minimum possible score for round N
            # because we are iterating `score_N` in increasing order (0, 1, ..., 100).
            ans = score_N
            break # No need to check higher scores, we found the minimum.

    # Print the result.
    # If a valid score_N was found, `ans` will hold that value.
    # Otherwise, `ans` will remain -1.
    print(ans)

# Call the solve function to execute the program logic.
if __name__ == '__main__':
    solve()