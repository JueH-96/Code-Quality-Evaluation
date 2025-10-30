import sys

def find_min_passengers():
    # Read input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Initialize minimum possible current number of passengers
    min_passengers = 0

    # Initialize current number of passengers
    current_passengers = 0

    # Iterate over each stop
    for i in range(N):
        # Update current number of passengers
        current_passengers += A[i]

        # Update minimum possible current number of passengers
        if current_passengers < min_passengers:
            min_passengers = current_passengers

    # If minimum possible current number of passengers is negative, set it to 0
    if min_passengers < 0:
        min_passengers = 0

    # Print the answer
    print(current_passengers - min_passengers)

# Call the function
find_min_passengers()