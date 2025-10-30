# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    T = []
    V = []
    index = 1
    for _ in range(N):
        T.append(int(data[index]))
        V.append(int(data[index+1]))
        index += 2
    
    current_water = 0
    prev_time = 0
    
    for i in range(N):
        current_time = T[i]
        # Calculate the time elapsed since the last event
        time_elapsed = current_time - prev_time
        # Reduce the water due to leakage
        current_water = max(0, current_water - time_elapsed)
        # Add the new water
        current_water += V[i]
        # Update the previous time
        prev_time = current_time
    
    print(current_water)

if __name__ == "__main__":
    main()