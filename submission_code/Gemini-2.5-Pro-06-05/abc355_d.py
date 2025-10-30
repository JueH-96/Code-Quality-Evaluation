# YOUR CODE HERE
import sys

def main():
    """
    Solves the interval intersection problem using a sweep-line algorithm.
    The program reads N intervals from standard input, calculates the number
    of pairs of intervals that intersect, and prints the result to standard output.
    """
    # Use a faster way to read input for large test cases
    input = sys.stdin.readline

    # Read the number of intervals
    try:
        n_str = input()
        # Handle empty input at the end of a file
        if not n_str:
            return
        N = int(n_str)
    except (ValueError, IndexError):
        # This handles cases with malformed input for N, though
        # problem constraints suggest input will be well-formed.
        return

    # Create a list of events from the intervals.
    # An event is a tuple (coordinate, type).
    # type = -1 represents the start of an interval (left endpoint).
    # type =  1 represents the end of an interval (right endpoint).
    events = []
    for _ in range(N):
        l, r = map(int, input().split())
        events.append((l, -1))
        events.append((r, 1))

    # Sort the events. Python's default tuple sort is lexicographical.
    # It sorts by coordinate first, then by type. Since -1 < 1, a 'start'
    # event at coordinate x will be processed before an 'end' event at the
    # same coordinate. This is crucial for correctly handling intervals that
    # meet at an endpoint (e.g., [1, 3] and [3, 5] should be counted as intersecting).
    events.sort()

    total_intersections = 0
    active_intervals = 0

    # Process events in sorted order (the "sweep-line").
    for _, event_type in events:
        if event_type == -1:  # This is a 'start' event.
            # A new interval begins. It intersects with all currently active intervals.
            # Add the current count of active intervals to the total.
            total_intersections += active_intervals
            # Increment the count of active intervals as this new one is now active.
            active_intervals += 1
        else:  # This is an 'end' event.
            # An interval ends. Decrement the count of active intervals.
            active_intervals -= 1

    # Print the final result.
    print(total_intersections)

if __name__ == "__main__":
    main()