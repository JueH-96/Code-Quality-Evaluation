def main():
    import sys
    input = sys.stdin.readline

    # Read input
    raw = input().split()
    if not raw:
        return
    N = int(raw[0])
    K = int(raw[1])
    
    groups = list(map(int, input().split()))
    
    # Simulation variables
    current_capacity = K
    start_count = 0

    # Process until no groups left
    while groups:
        # If the current capacity is less than the number of people in the front group,
        # start the attraction, reset the capacity, and increase start count.
        if current_capacity < groups[0]:
            start_count += 1
            current_capacity = K
        else:
            # Otherwise, let the group board and decrease the capacity
            current_capacity -= groups.pop(0)
    
    # Once the queue is empty, start the attraction one final time.
    start_count += 1
    
    # Output the result.
    print(start_count)

if __name__ == '__main__':
    main()