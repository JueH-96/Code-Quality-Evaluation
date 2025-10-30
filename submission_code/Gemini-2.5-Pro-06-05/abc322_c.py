# YOUR CODE HERE
import sys

def solve():
    """
    Solves the AtCoder Festival problem efficiently.
    """
    # Read N and M from standard input.
    try:
        N, M = map(int, sys.stdin.readline().split())
    except ValueError:
        # Handles potential empty input lines, though not expected by constraints.
        return

    # Read the M firework days into a list A.
    A = list(map(int, sys.stdin.readline().split()))

    # `firework_day_ptr` points to the index in A of the next upcoming firework day.
    # It starts at 0, pointing to the first firework day, A_1.
    firework_day_ptr = 0

    # A list to store results for efficient printing at the end.
    answers = []

    # Iterate through each day of the festival, from 1 to N.
    for current_day in range(1, N + 1):
        # Advance the pointer to find the first firework day on or after the current day.
        # This while loop ensures `A[firework_day_ptr]` is always the earliest
        # firework day that is >= `current_day`.
        # The total number of increments of `firework_day_ptr` across all iterations
        # of the outer loop is at most M.
        while A[firework_day_ptr] < current_day:
            firework_day_ptr += 1
            
        # The upcoming firework day for `current_day`.
        next_firework_day = A[firework_day_ptr]
        
        # Calculate the wait time and append it to the answers list as a string.
        wait_days = next_firework_day - current_day
        answers.append(str(wait_days))

    # Print all the collected answers, separated by newlines.
    # This is typically faster than printing in a loop for a large number of lines.
    print("
".join(answers))

solve()