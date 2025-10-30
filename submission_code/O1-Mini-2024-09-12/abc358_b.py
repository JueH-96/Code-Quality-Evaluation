# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read().split()
    N, A = int(input[0]), int(input[1])
    T = list(map(int, input[2:2+N]))
    
    available_time = 0
    for t in T:
        start_time = max(t, available_time)
        finish_time = start_time + A
        available_time = finish_time
        print(finish_time)

if __name__ == "__main__":
    main()