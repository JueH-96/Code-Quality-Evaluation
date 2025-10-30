import sys

# YOUR CODE HERE
def solve():
    """
    Reads problem inputs, calculates the minimum number of days, and prints the result.
    """
    try:
        # Read the first line containing N, T, and P.
        # N: number of people
        # T: target hair length
        # P: target number of people
        n, t, p = map(int, sys.stdin.readline().split())

        # Read the second line containing the initial hair lengths of N people.
        # The problem constraints (N >= 1) ensure this line exists.
        l = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # This handles potential empty input or malformed lines, though
        # not expected under the problem's specified constraints.
        return

    # For each person, calculate the number of days they need for their hair to reach length T.
    # If a person's hair length `h` is already T or more, they need 0 days.
    # Otherwise, they need `T - h` days.
    # This logic is captured by `max(0, T - h)`.
    days_needed = [max(0, t - h) for h in l]

    # Sort the list of required days in non-decreasing order.
    days_needed.sort()

    # To find the first day when at least P people meet the criteria, we need to
    # wait long enough for the P-th fastest person to reach the target length.
    # In our sorted list of `days_needed`, this corresponds to the P-th element.
    # In a 0-indexed list, this is the element at index `p - 1`.
    # The constraint `1 <= P <= N` guarantees that this index is always valid.
    answer = days_needed[p - 1]

    # Print the final answer to standard output.
    print(answer)

# Execute the solution
solve()