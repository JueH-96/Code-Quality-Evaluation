def main():
    import sys
    input_data = sys.stdin.read().split()
    # First element is N, the number of water additions.
    n = int(input_data[0])
    water = 0
    prev_time = 0
    index = 1
    for i in range(n):
        # For the i-th event, get time T and volume V.
        t = int(input_data[index])
        v = int(input_data[index + 1])
        index += 2
        
        # Calculate the time difference from the previous event.
        dt = t - prev_time
        
        # The humidifier leaks 1 liter per unit time, but it can't go below 0.
        water = max(water - dt, 0)
        
        # Add water at time t.
        water += v
        
        # Update the previous time marker.
        prev_time = t
    
    print(water)

# Calling main function
if __name__ == "__main__":
    main()