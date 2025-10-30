# YOUR CODE HERE
import sys

def solve():
    # Read N (number of groups) and K (capacity) from the first line of stdin
    line1 = sys.stdin.readline().split()
    n = int(line1[0])
    k = int(line1[1])

    # Constraints state N >= 1, so we expect at least one group.
    
    # Read the list of group sizes A from the second line of stdin
    # The list will contain N integers.
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize the count of started attractions (rides)
    # We start the count at 1. This represents the first ride being prepared 
    # before processing any groups. Subsequent increments occur only when a new 
    # ride is needed because the current one cannot accommodate the next group.
    ride_count = 1
    
    # Initialize the remaining capacity of the current ride
    # Initially, the first ride has full capacity K.
    current_capacity = k

    # Iterate through each group size in the list A
    for group_size in a:
        # Check if the current group fits in the remaining capacity of the current ride
        if current_capacity >= group_size:
            # If it fits, the group boards the current ride.
            # Reduce the remaining capacity of the current ride.
            current_capacity -= group_size
        else:
            # If the group does not fit, a new ride must be started.
            ride_count += 1
            
            # The new ride starts with full capacity K.
            current_capacity = k
            
            # The current group (which couldn't fit in the previous ride) 
            # boards this new ride. Since the constraint A_i <= K holds, 
            # any single group is guaranteed to fit in a fresh ride.
            current_capacity -= group_size

    # After iterating through all groups, ride_count holds the total number
    # of times the attraction was started. Print this count to standard output.
    print(ride_count)

# Execute the main logic of the program by calling the solve function
solve()