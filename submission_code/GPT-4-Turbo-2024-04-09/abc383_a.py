# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    events = []
    
    index = 1
    for _ in range(N):
        T = int(data[index])
        V = int(data[index + 1])
        events.append((T, V))
        index += 2
    
    # Initial conditions
    current_water = 0
    last_time = 0
    
    for time, volume in events:
        # Calculate water loss due to leakage
        time_elapsed = time - last_time
        current_water = max(0, current_water - time_elapsed)
        
        # Add new volume of water
        current_water += volume
        
        # Update last_time to current event time
        last_time = time
    
    # Output the final amount of water after the last addition
    print(current_water)

if __name__ == "__main__":
    main()