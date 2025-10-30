import sys

def solve():
    """
    Reads input, solves the bus passenger problem, and prints the result.
    """
    # Read N from the first line of stdin.
    # While N is given, we can also infer it from the length of the list `a`.
    try:
        n_line = sys.stdin.readline()
        if not n_line:
            return
        n = int(n_line)

        # Read the N space-separated integers for passenger changes.
        a = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # Handle potential malformed input.
        return

    # The goal is to find the minimum possible number of passengers at the end of the trip,
    # consistent with the constraint that the passenger count is always non-negative.
    
    # Let F be the final number of passengers. We work backwards to determine constraints on F.
    # At any point, the number of passengers must be non-negative. This implies that F must be
    # greater than or equal to the sum of passenger changes from that point to the end.
    # F must also be >= 0. Therefore, the minimum F is the maximum of 0 and all suffix sums.

    # `max_required_at_end` will store the maximum suffix sum encountered,
    # which is the minimum possible value for the final number of passengers.
    # It's initialized to 0 to handle the F >= 0 constraint.
    max_required_at_end = 0

    # `current_suffix_sum` will hold the sum of elements from the current position to the end.
    current_suffix_sum = 0

    # Iterate from the last stop to the first to compute suffix sums efficiently.
    for i in range(n - 1, -1, -1):
        # Add the current stop's change to the running suffix sum.
        current_suffix_sum += a[i]
        
        # The final number of passengers must be at least as large as the current
        # suffix sum to ensure the passenger count was non-negative before this point.
        # We update our required minimum for F if the current suffix sum is larger.
        if current_suffix_sum > max_required_at_end:
            max_required_at_end = current_suffix_sum

    # Print the final result.
    print(max_required_at_end)

solve()