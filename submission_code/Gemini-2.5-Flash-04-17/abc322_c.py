# YOUR CODE HERE
import sys

def solve():
    # Read N (total days) and M (number of fireworks days) from the first line.
    # N, M are integers.
    N, M = map(int, sys.stdin.readline().split())

    # Read the list of fireworks days A from the second line.
    # A contains M integers, which are 1-indexed day numbers.
    # The list A is guaranteed to be sorted in strictly increasing order: A_1 < A_2 < ... < A_M.
    # It is also guaranteed that the last day has fireworks: A_M = N.
    A = list(map(int, sys.stdin.readline().split()))

    # Create a list to store the results for each day of the festival.
    # We want results for days 1 through N. Using a list of size N+1 allows for
    # 1-based indexing convenience, where results[i] stores the answer for day i.
    results = [0] * (N + 1)

    # Initialize an index to track the current position in the sorted list A.
    # fireworks_idx will always point to the first element A[fireworks_idx]
    # such that A[fireworks_idx] >= the current day 'i' we are processing.
    # Initially, for day i=1, the first fireworks day >= 1 is A[0] (since A_1 >= 1).
    fireworks_idx = 0

    # Iterate through each day of the festival, from day 1 to day N.
    for i in range(1, N + 1):
        # Advance the fireworks_idx pointer to find the first fireworks day A[fireworks_idx]
        # that is greater than or equal to the current day 'i'.
        # We check `fireworks_idx < M` to make sure we don't go out of bounds of list A.
        # The second condition `A[fireworks_idx] < i` means the current fireworks day
        # A[fireworks_idx] is before day 'i', so we need to look at the next fireworks day.
        # The loop continues advancing fireworks_idx until A[fireworks_idx] >= i.
        # Since A[M-1] = N and our loop for 'i' goes only up to N, the condition A[fireworks_idx] < i
        # will eventually become false before `fireworks_idx` reaches M because A[M-1] = N is
        # always >= i for any i in the range [1, N]. Thus, the `while` loop is guaranteed
        # to terminate with `fireworks_idx < M`, making `A[fireworks_idx]` a valid access.
        while fireworks_idx < M and A[fireworks_idx] < i:
            fireworks_idx += 1

        # After the while loop finishes, A[fireworks_idx] is the smallest fireworks day
        # in the list A that is greater than or equal to the current day 'i'.
        # This is exactly the first day with fireworks on or after day 'i'.

        # The problem asks for the number of days *later* from day 'i'.
        # If the first fireworks day is F, the number of days later is F - i.
        # Here, F is A[fireworks_idx].
        days_later = A[fireworks_idx] - i

        # Store the calculated number of days later for day 'i' in our results list.
        results[i] = days_later

    # After calculating the results for all days from 1 to N, print them.
    # Each result should be printed on a new line.
    # Using sys.stdout.write is generally more efficient for printing a large number
    # of lines compared to repeated calls to the standard print() function.
    for i in range(1, N + 1):
        sys.stdout.write(str(results[i]) + '
')

# Call the solve function to run the program logic.
solve()