def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    values = data[1:]

    water = 0
    prev_time = 0

    for i in range(N):
        T = int(values[2*i])
        V = int(values[2*i + 1])
        
        # Decrease water based on the time elapsed
        elapsed = T - prev_time
        water = max(0, water - elapsed)
        
        # Add the water
        water += V
        
        # Update the current time
        prev_time = T

    print(water)

# Do not forget to call the main function
if __name__ == "__main__":
    main()