# YOUR CODE HERE
import sys

# Read N (number of people) and A (purchase time per person) from the first line
line1 = sys.stdin.readline().split()
N = int(line1[0])
A = int(line1[1])

# Read the arrival times T for each person from the second line
# The arrival times are given in increasing order, which corresponds to the order of people 1 to N.
line2 = sys.stdin.readline().split()
T = list(map(int, line2))

# Initialize the time the ticket booth becomes free.
# Initially, there is no one in line, so the booth is free at time 0.
# This variable will store the finish time of the *previous* person,
# which is the earliest time the *current* person can potentially start,
# assuming they have already arrived.
booth_free_time = 0

# Iterate through each person from 1 to N (using index i from 0 to N-1)
for i in range(N):
    # Get the arrival time of the current person (the (i+1)-th person)
    arrival_time = T[i]

    # Determine when the current person starts purchasing.
    # A person can only start purchasing if both conditions are met:
    # 1. They have arrived (at or after their arrival_time).
    # 2. The ticket booth is free (at or after the booth_free_time, which is
    #    when the previous person finished).
    # Therefore, the start time is the later of their arrival time and the time the booth became free.
    start_time = max(arrival_time, booth_free_time)

    # Calculate the finish time for the current person.
    # The purchase process takes A seconds, starting from their start_time.
    finish_time = start_time + A

    # Print the finish time for the current person.
    # Each finish time should be printed on a new line.
    print(finish_time)

    # Update the time the booth becomes free for the next person.
    # The booth becomes free exactly when the current person finishes.
    booth_free_time = finish_time
# YOUR CODE HERE