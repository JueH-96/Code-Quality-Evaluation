# YOUR CODE HERE
import sys

def solve():
    """
    Reads interval data from stdin, calculates the number of intersecting pairs
    using a sweep-line algorithm, and prints the result to stdout.
    """
    # Read the number of intervals N from the first line of input.
    n = int(sys.stdin.readline())
    
    # Create a list to store events. Each event will be represented as a tuple:
    # (coordinate, type)
    # 'coordinate' is the x-value where the event occurs (either l_i or r_i).
    # 'type' indicates whether it's a start (0) or end (1) of an interval.
    # Using 0 for start and 1 for end ensures that when coordinates are equal,
    # start events are processed before end events due to standard tuple sorting.
    events = []
    
    # Read the N intervals and populate the events list.
    # Each interval [l_i, r_i] generates two events: a start event at l_i and an end event at r_i.
    for _ in range(n):
        # Read the left (l_i) and right (r_i) endpoints of the current interval.
        l, r = map(int, sys.stdin.readline().split())
        
        # Add the start event: (coordinate l_i, type 0)
        events.append((l, 0)) 
        # Add the end event: (coordinate r_i, type 1)
        events.append((r, 1)) 

    # Sort the events list. The sorting criteria are:
    # 1. Primarily by coordinate, in ascending order.
    # 2. Secondarily by type, in ascending order (0 comes before 1).
    # This ensures that we process events from left to right along the number line,
    # and at any given coordinate, all interval starts are processed before any interval ends.
    # The time complexity of sorting 2N events is O(N log N).
    events.sort()

    # Initialize the total count of intersecting pairs. This will be the final answer.
    intersection_count = 0
    
    # Initialize the count of intervals that are currently "active".
    # An interval is considered active if the sweep line has passed its start point (l_i)
    # but has not yet reached its end point (r_i).
    active_intervals = 0

    # Iterate through the sorted events, simulating a sweep line moving across the number line.
    # This loop processes 2N events, taking O(N) time in total.
    for event in events:
        # Unpack the coordinate and type of the current event.
        # The coordinate itself isn't explicitly needed in the core logic below,
        # but it dictates the order of processing.
        # coord = event[0] 
        type = event[1]  # type is 0 for start, 1 for end

        if type == 0: # This event marks the start of an interval.
            # When a new interval starts, it intersects with all intervals that are currently active.
            # The number of currently active intervals is `active_intervals`.
            # We add this number to our total count of intersections. Each active interval
            # forms an intersecting pair with the interval that just started.
            intersection_count += active_intervals
            
            # Increment the count of active intervals because the interval corresponding to this
            # start event is now considered active.
            active_intervals += 1
        else: # This event marks the end of an interval.
            # When an interval ends, it is no longer active.
            # Decrement the count of active intervals.
            # Note: Because every interval has l_i < r_i and we process starts before ends
            # at the same coordinate, `active_intervals` will always be at least 1 before
            # this decrement when processing an end event corresponding to a valid interval.
            active_intervals -= 1

    # After iterating through all events, `intersection_count` holds the total number
    # of pairs of intervals (i, j) with i < j such that interval i and interval j intersect.
    # Print the final result to standard output.
    print(intersection_count)

# Call the solve function to execute the main logic of the program.
solve()