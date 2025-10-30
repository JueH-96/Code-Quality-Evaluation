# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    events = [tuple(map(int, line.split())) for line in data[1:]]
    
    current_water = 0
    current_time = 0
    
    for T, V in events:
        # Calculate water remaining before adding new water
        current_water = max(0, current_water - (T - current_time))
        # Add new water
        current_water += V
        # Update current time
        current_time = T
    
    print(current_water)

if __name__ == "__main__":
    main()