# Input processing
N, A = map(int, input().split())
T = list(map(int, input().split()))

# Variable to keep track of when the ticket booth becomes available
current_time = 0

# Process each person
for arrival_time in T:
    # If the person arrives after the booth is available, they start immediately
    # Otherwise, they wait until the booth becomes available
    current_time = max(current_time, arrival_time)
    
    # Add the purchase time
    current_time += A
    
    # Print when this person finishes
    print(current_time)