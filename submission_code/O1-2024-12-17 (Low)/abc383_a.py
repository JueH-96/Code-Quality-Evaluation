def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    data_pairs = input_data[1:]
    current_water = 0
    last_time = 0
    idx = 0
    
    for _ in range(N):
        T = int(data_pairs[idx])
        V = int(data_pairs[idx + 1])
        idx += 2
        # Decrease water by the time difference since last addition
        current_water = max(0, current_water - (T - last_time))
        # Add new water
        current_water += V
        # Update the last time
        last_time = T

    print(current_water)

# Call main function
if __name__ == "__main__":
    main()