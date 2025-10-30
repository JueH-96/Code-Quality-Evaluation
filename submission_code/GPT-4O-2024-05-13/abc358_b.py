# YOUR CODE HERE
import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))
    
    current_time = 0
    results = []
    
    for i in range(N):
        arrival_time = T[i]
        if current_time < arrival_time:
            current_time = arrival_time
        current_time += A
        results.append(current_time)
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()