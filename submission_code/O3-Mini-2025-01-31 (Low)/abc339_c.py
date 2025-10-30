def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
        
    n = int(input_data[0])
    # Read the list of changes
    changes = list(map(int, input_data[1:]))
    
    current_passengers = 0
    min_passengers = 0  # tracks the lowest passenger count seen
    
    # Simulate the stops and record the minimum passenger count
    for change in changes:
        current_passengers += change
        if current_passengers < min_passengers:
            min_passengers = current_passengers
    
    # To ensure the minimum count is non-negative, we can add an offset if needed.
    # If min_passengers is negative, offset will be -min_passengers; otherwise zero.
    offset = -min_passengers if min_passengers < 0 else 0
    
    # The current number of passengers when starting with the adjusted initial count
    answer = offset + current_passengers
    
    print(answer)

if __name__ == '__main__':
    main()