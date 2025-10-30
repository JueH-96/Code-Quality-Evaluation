# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    times = []
    volumes = []
    
    for i in range(N):
        times.append(int(data[2 * i + 1]))
        volumes.append(int(data[2 * i + 2]))
    
    current_water = 0
    current_time = 0
    
    for i in range(N):
        time_passed = times[i] - current_time
        current_water = max(0, current_water - time_passed)
        current_water += volumes[i]
        current_time = times[i]
    
    print(current_water)

if __name__ == "__main__":
    main()