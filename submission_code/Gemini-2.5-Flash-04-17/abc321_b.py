# YOUR CODE HERE
import sys

# Read N and X from the first line of standard input
line1 = sys.stdin.readline().split()
N = int(line1[0])
X = int(line1[1])

# Read the N-1 scores A_1, ..., A_{N-1} from the second line
line2 = sys.stdin.readline().split()
A = [int(a) for a in line2]

# To efficiently calculate the min and max of the initial N-1 scores,
# we sort the list A. Sorting takes O(N log N) time.
# This sorted list will be used to find the minimum and maximum among the initial scores.
A_sorted = sorted(A)

# Calculate the sum of the initial N-1 scores. This takes O(N) time.
sum_A_initial = sum(A)

# Get the minimum and maximum scores from the initial N-1 rounds.
# A_sorted has length N-1, so the indices are 0 to N-2.
# O(1) operation after sorting.
# Note: These are the min/max *among the first N-1 scores*.
min_A_initial = A_sorted[0]
max_A_initial = A_sorted[N-2]

# We need to find the minimum integer score s_n (0 <= s_n <= 100)
# for the N-th round such that the final grade is X or higher.
# The final grade depends on s_n. We iterate through all possible integer values of s_n
# from 0 to 100 in increasing order. The first s_n that satisfies the condition
# will be the minimum required score because the final grade is a non-decreasing
# function of s_n within the range [0, 100].
found_score = False
for s_n in range(0, 101): # Iterate through possible scores 0, 1, ..., 100
    # Consider the set of N scores: A_1, ..., A_{N-1}, s_n.
    # The sum of these N scores is the sum of initial scores plus s_n.
    # O(1) operation.
    sum_total_N_scores = sum_A_initial + s_n

    # The minimum score among all N scores (initial N-1 plus s_n)
    # is the minimum of the minimum of the initial scores and s_n.
    # O(1) operation.
    min_among_N_scores = min(min_A_initial, s_n)

    # The maximum score among all N scores (initial N-1 plus s_n)
    # is the maximum of the maximum of the initial scores and s_n.
    # O(1) operation.
    max_among_N_scores = max(max_A_initial, s_n)

    # The final grade is the sum of N-2 scores excluding the highest and lowest from the set of N scores.
    # This is equivalent to the total sum of N scores minus the lowest and highest scores among the N scores.
    # O(1) operation.
    final_grade = sum_total_N_scores - min_among_N_scores - max_among_N_scores

    # Check if the calculated grade meets the target X.
    if final_grade >= X:
        # We found the minimum score s_n that works because we are iterating in increasing order (0 to 100).
        # Print the score and set the flag.
        print(s_n)
        found_score = True
        # Exit the loop immediately since we found the minimum required score.
        break

# If the loop finishes iterating through all possible scores (0-100) without finding any s_n
# that results in a grade >= X, it means it's impossible to achieve the target grade X.
# The 'found_score' flag will still be False in this case.
if not found_score:
    # Print -1 as it's impossible to reach the target grade.
    print(-1)