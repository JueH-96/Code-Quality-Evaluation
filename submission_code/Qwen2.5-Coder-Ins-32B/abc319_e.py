# YOUR CODE HERE
import sys
import math

def main():
    import sys
    import math
    input = sys.stdin.read().split()
    index = 0
    
    N = int(input[index])
    X = int(input[index + 1])
    Y = int(input[index + 2])
    index += 3
    
    P = []
    T = []
    for _ in range(N - 1):
        P.append(int(input[index]))
        T.append(int(input[index + 1]))
        index += 2
    
    Q = int(input[index])
    index += 1
    queries = [int(input[index + i]) for i in range(Q)]
    
    def next_bus_time(current_time, period):
        if current_time % period == 0:
            return current_time
        return (current_time // period + 1) * period
    
    def earliest_arrival_time(q):
        current_time = q + X
        for i in range(N - 1):
            current_time = next_bus_time(current_time, P[i]) + T[i]
        return current_time + Y
    
    results = [earliest_arrival_time(q) for q in queries]
    
    for result in results:
        print(result)

if __name__ == "__main__":
    main()