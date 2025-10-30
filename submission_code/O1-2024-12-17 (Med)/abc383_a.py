def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    
    # We'll keep track of the current time of the last addition
    prev_time = 0
    # We'll keep track of the current volume of water
    volume = 0
    
    idx = 1  # Index to read T_i and V_i from data
    for _ in range(N):
        T_i = int(data[idx]); V_i = int(data[idx+1])
        idx += 2
        
        # Decrease water over the time difference (if volume was positive)
        time_diff = T_i - prev_time
        volume = max(0, volume - time_diff)
        
        # Add water
        volume += V_i
        
        # Update previous time
        prev_time = T_i
    
    # Print final volume
    print(volume)

# Do not forget to call the main function
if __name__ == "__main__":
    main()