# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N, A, and arrival times T.
    Calculates and prints the finish time for each person purchasing a ticket
    at a single booth.
    """
    # Read N (number of people) and A (processing time per person) from the first line of input.
    # The variable N indicates the number of arrival times that follow, but it's not explicitly 
    # needed in the iteration logic if we iterate directly over the list of arrival times.
    n, a = map(int, sys.stdin.readline().split()) 
    
    # Read the list of arrival times T from the second line of input.
    # The input guarantees that T is sorted in ascending order: T_1 < T_2 < ... < T_N.
    t = list(map(int, sys.stdin.readline().split())) 

    # Initialize the time when the ticket booth becomes free.
    # Before the first person arrives, the booth is considered free at time 0.
    last_finish_time = 0

    # Iterate through each person's arrival time in the list T.
    for arrival_time in t:
        # Determine the time the current person can start purchasing their ticket.
        # A person can only start after they have arrived AND after the booth is free 
        # (i.e., the previous person has finished).
        # Therefore, the start time is the maximum of their arrival time and the time 
        # the booth became free after the previous person finished.
        start_time = max(arrival_time, last_finish_time)

        # Calculate the time the current person finishes purchasing their ticket.
        # This is simply their start time plus the fixed processing time A.
        finish_time = start_time + a

        # Output the calculated finish time for the current person.
        # The problem requires printing the finish time for each person on a new line.
        print(finish_time)

        # Update the time the booth becomes free for the next person.
        # After the current person finishes, the booth becomes free at `finish_time`.
        # This value will be used in the calculation for the next person in the iteration.
        last_finish_time = finish_time

# Execute the main logic by calling the solve function.
# This reads the input, processes it, and prints the output as required.
solve()
# YOUR CODE HERE