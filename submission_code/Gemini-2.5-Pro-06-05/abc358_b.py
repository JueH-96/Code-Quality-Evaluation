import sys

def solve():
    """
    Reads input from stdin, simulates the ticket booth process,
    and prints the finish time for each person to stdout.
    """
    try:
        # Read the number of people (N) and the fixed purchase time (A).
        n, a = map(int, sys.stdin.readline().split())

        # Read the list of arrival times for each person.
        t_arrivals = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # This handles cases where the input stream is empty,
        # which is not expected under the problem's constraints.
        return

    # This variable tracks the time the ticket booth becomes available for the next person.
    # Initially, at time 0, the booth is free.
    booth_free_time = 0

    # Process each person in their given arrival order.
    for arrival_time in t_arrivals:
        # A person can start buying their ticket at the maximum of two moments:
        # 1. Their own arrival time (`arrival_time`). They must be present to start.
        # 2. The time the booth becomes free (`booth_free_time`). They must wait if the booth is busy.
        start_time = max(arrival_time, booth_free_time)
        
        # The time this person will finish is their start time plus the processing time A.
        finish_time = start_time + a
        
        # Print the calculated finish time for the current person.
        print(finish_time)
        
        # After this person is done, the booth becomes free at their `finish_time`.
        # This updated time is used to determine the start time for the next person.
        booth_free_time = finish_time

solve()