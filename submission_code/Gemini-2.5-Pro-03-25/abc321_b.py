# YOUR CODE HERE
import sys

def solve():
    """
    Reads the input, calculates the minimum score needed in the Nth round
    to achieve a final grade of at least X, and prints the result.
    """
    # Read N (total number of rounds) and X (target final grade) from the first line of input.
    # Input values are space-separated integers.
    n, x = map(int, sys.stdin.readline().split())

    # Read the N-1 scores (A_1, A_2, ..., A_{N-1}) obtained in the first N-1 rounds.
    # Input values are space-separated integers on the second line.
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize the variable to store the minimum required score for the Nth round.
    # Set it to -1 initially, indicating that it's potentially impossible to reach the target grade.
    min_required_s = -1

    # Iterate through all possible integer scores 's' for the Nth round.
    # According to the problem statement, the score must be between 0 and 100, inclusive.
    # The range(101) generates integers from 0 up to 100.
    for s in range(101): 
        
        # Create a list containing all N scores.
        # This includes the N-1 scores already obtained (list 'a') and the current hypothetical score 's' for the Nth round.
        # We create a new list `all_scores` in each iteration by concatenating `a` and `[s]`.
        all_scores = a + [s]

        # Sort the list of N scores in non-decreasing (ascending) order.
        # This is necessary because the final grade calculation depends on the sorted order
        # to exclude the minimum and maximum scores.
        all_scores.sort()

        # Calculate the final grade based on the sorted list of N scores.
        # The final grade is the sum of the scores excluding the lowest score (at index 0)
        # and the highest score (at index N-1, which is the last element).
        # This means summing the elements from index 1 up to index N-2 (inclusive).
        # Python's list slicing `[1:-1]` extracts elements from index 1 up to, but not including,
        # the last element, effectively selecting indices 1, 2, ..., N-2.
        # The `sum()` function calculates the sum of these selected middle scores.
        current_grade = sum(all_scores[1:-1])

        # Check if the calculated grade (`current_grade`) meets or exceeds the target grade `x`.
        if current_grade >= x:
            # If the condition is met, it means that achieving a score of 's' in the Nth round
            # results in a final grade of at least X.
            # Since we are iterating 's' in increasing order (starting from 0),
            # the first value of 's' for which this condition holds true is the minimum possible score required.
            min_required_s = s
            
            # We have found the minimum required score, so there's no need to check higher scores.
            # We can exit the loop early using 'break'.
            break

    # After the loop finishes (either because we found the minimum score and used 'break',
    # or because we checked all possible scores from 0 to 100 without meeting the condition):
    # Print the value stored in `min_required_s`.
    # If a suitable score was found, it will hold that minimum score (an integer between 0 and 100).
    # If no score between 0 and 100 resulted in a final grade of at least X,
    # `min_required_s` will retain its initial value of -1, indicating impossibility.
    print(min_required_s)

# Execute the solve function to run the main logic of the program.
solve()