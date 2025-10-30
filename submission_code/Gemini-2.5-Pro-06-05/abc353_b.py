# YOUR CODE HERE
import sys

def solve():
    """
    Reads input from stdin, simulates the attraction guidance process,
    and prints the total number of times the attraction is started.
    """
    # Read the number of groups (N) and the attraction's capacity (K).
    try:
        # Read the first line which contains N and K.
        line1 = sys.stdin.readline()
        # If the line is empty (end of input), do nothing.
        if not line1:
            return
        n, k = map(int, line1.split())
    except (IOError, ValueError):
        # Gracefully handle potential empty lines or format errors.
        return

    # According to the problem constraints, N >= 1.
    # If N were 0, there would be no groups and thus no rides.
    if n == 0:
        print(0)
        return

    # Read the sizes of the N groups from the second line.
    groups = list(map(int, sys.stdin.readline().split()))
    
    # Initialize the count of rides started. Since N >= 1, at least one ride
    # is necessary to accommodate the first group(s).
    rides_started = 1
    
    # Initialize the capacity for the first ride.
    current_capacity = k
    
    # Iterate through each group in the queue.
    for group_size in groups:
        # Check if the current group can fit in the remaining seats.
        if current_capacity >= group_size:
            # If they fit, board them and reduce the available capacity.
            current_capacity -= group_size
        else:
            # If they do not fit, a new ride must be started.
            rides_started += 1
            # This new ride starts with full capacity K, and the current group
            # is immediately boarded. The constraint A_i <= K ensures they will fit.
            current_capacity = k - group_size
            
    # Print the final count to standard output.
    print(rides_started)

# Execute the solution
solve()