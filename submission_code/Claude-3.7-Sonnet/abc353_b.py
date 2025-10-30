def solve():
    N, K = map(int, input().split())
    A = list(map(int, input().split()))
    
    starts = 0  # Counter for attraction starts
    empty_seats = K  # Initially, all seats are empty
    
    i = 0  # Index for groups in the queue
    while i < N:
        if A[i] > empty_seats:  # If not enough seats for the group
            starts += 1  # Start the attraction
            empty_seats = K  # Reset the number of empty seats
        else:
            empty_seats -= A[i]  # Guide the group to the attraction, reducing empty seats
            i += 1  # Move to the next group
    
    # After all groups are processed, start the attraction once more
    starts += 1
    
    return starts

if __name__ == "__main__":
    print(solve())