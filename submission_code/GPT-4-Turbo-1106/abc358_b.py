# Read the input values
N, A = map(int, input().split())
T = list(map(int, input().split()))

# Initialize the time when the last person finished
last_finish_time = 0

# Process each person in the queue
for arrival_time in T:
    # Calculate the start time for the current person
    start_time = max(arrival_time, last_finish_time)
    # Calculate the finish time for the current person
    finish_time = start_time + A
    # Update the last finish time
    last_finish_time = finish_time
    # Output the finish time for the current person
    print(finish_time)