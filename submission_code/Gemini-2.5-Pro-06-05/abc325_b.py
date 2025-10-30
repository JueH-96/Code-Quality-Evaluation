# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, calculates the maximum number of employees that can attend a meeting,
    and prints the result.
    """
    try:
        # Read the number of bases.
        # An empty line might be read at the end of input in some environments.
        line = sys.stdin.readline()
        if not line.strip():
            return
        N = int(line)
    except (ValueError, IndexError):
        # If input is invalid, we can't proceed.
        # Assuming N=0 for this case, the answer would be 0.
        print(0)
        return

    # `participants_per_hour[T]` will store the total number of employees
    # who can attend a meeting starting at T:00 UTC.
    participants_per_hour = [0] * 24

    # Process the data for each of the N bases.
    for _ in range(N):
        W, X = map(int, sys.stdin.readline().split())

        # Employees at a base can participate if the meeting starts locally
        # between 9:00 and 17:00. This is a 9-hour window.
        # We find the corresponding UTC time for the start of this window (local time 9:00).
        # local_time = (utc_time + X) % 24
        # => utc_time = (local_time - X + 24) % 24
        utc_window_start = (9 - X + 24) % 24

        # For each of the 9 hours in the valid window, we add the number of
        # employees (W) to the total for that UTC start hour.
        for i in range(9):
            utc_hour = (utc_window_start + i) % 24
            participants_per_hour[utc_hour] += W

    # The maximum number of participants is the highest total found across all
    # possible UTC start hours.
    print(max(participants_per_hour))

solve()