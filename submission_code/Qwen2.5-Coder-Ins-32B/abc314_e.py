# YOUR CODE HERE
import sys
import numpy as np

def main():
    import sys
    import numpy as np
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    data = input[2:]
    
    wheels = []
    index = 0
    for _ in range(N):
        C = int(data[index])
        P = int(data[index + 1])
        S = list(map(int, data[index + 2:index + 2 + P]))
        wheels.append((C, P, S))
        index += 2 + P
    
    # Initialize the expected cost array
    exp_cost = np.zeros(M + 1)
    
    # Fill the expected cost array from M-1 to 0
    for points in range(M - 1, -1, -1):
        min_cost = float('inf')
        for C, P, S in wheels:
            expected_points = sum(S) / P
            if expected_points == 0:
                continue
            # Calculate the expected cost for this wheel
            cost = C
            for s in S:
                if points + s < M:
                    cost += exp_cost[points + s] * (S.count(s) / P)
            min_cost = min(min_cost, cost)
        exp_cost[points] = min_cost
    
    print(exp_cost[0])

if __name__ == "__main__":
    main()