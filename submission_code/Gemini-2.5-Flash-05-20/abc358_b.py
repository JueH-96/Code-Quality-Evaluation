import sys

def solve():
    # Read N and A from the first line of input
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    A = int(line1[1])

    # Read the list of arrival times T from the second line
    T = list(map(int, sys.stdin.readline().split()))

    # Initialize current_service_finish_time.
    # This variable keeps track of the exact time when the ticket booth becomes free
    # after the last person (or current person) finishes their purchase.
    # Initially, the booth is empty and free at time 0.
    current_service_finish_time = 0

    # Iterate through each person from 0 to N-1
    for i in range(N):
        person_arrival_time = T[i]

        # Determine when the current person starts their purchase.
        # They start either when they arrive (if the booth is free)
        # or when the previous person finishes (if the booth is busy).
        # This is the maximum of their arrival time and the time the booth becomes free.
        service_start_time = max(person_arrival_time, current_service_finish_time)

        # Calculate when the current person finishes their purchase.
        # This is their start time plus the duration A.
        person_finish_time = service_start_time + A

        # Update current_service_finish_time for the next person.
        # The booth will be busy until this person finishes.
        current_service_finish_time = person_finish_time

        # Print the finish time for the current person.
        print(person_finish_time)

# Call the solve function to run the program
solve()