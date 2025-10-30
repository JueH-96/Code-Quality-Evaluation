# YOUR CODE HERE
def amusement_park(N, K, A):
    # Initialize the number of times the attraction is started
    attraction_starts = 0
    # Initialize the number of empty seats
    empty_seats = K

    # Process each group in the queue
    for group in A:
        if group > empty_seats:
            # If the current group cannot fit, start the attraction
            attraction_starts += 1
            # Reset the empty seats
            empty_seats = K
        
        # Guide the current group to the attraction
        empty_seats -= group

    # If there are any remaining groups that have been guided but not started, start the attraction one more time
    if empty_seats < K:
        attraction_starts += 1

    return attraction_starts

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    result = amusement_park(N, K, A)
    print(result)