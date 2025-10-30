import sys

def main():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))

    rides_started = 1  # Accounts for the first/final ride cycle.
                       # Since N >= 1, at least one ride will occur.
    
    current_empty_seats = K  # Represents empty seats in the current ride being filled.

    for group_size in A:
        # Try to fit the current group into the current ride.
        if current_empty_seats < group_size:
            # Not enough space. The current ride must depart.
            # A new ride cycle starts for the current group.
            rides_started += 1
            current_empty_seats = K  # Reset to full capacity for the new ride cycle.
        
        # The current group boards.
        # Since A_i <= K, and current_empty_seats is K if it was just reset,
        # the group is guaranteed to fit. current_empty_seats will be >= 0.
        current_empty_seats -= group_size
    
    print(rides_started)

if __name__ == '__main__':
    main()