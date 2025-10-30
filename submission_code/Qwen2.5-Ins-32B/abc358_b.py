# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))
    
    finish_times = []
    current_time = 0
    
    for i in range(N):
        if T[i] > current_time:
            current_time = T[i]
        current_time += A
        finish_times.append(current_time)
    
    for time in finish_times:
        print(time)

if __name__ == "__main__":
    main()