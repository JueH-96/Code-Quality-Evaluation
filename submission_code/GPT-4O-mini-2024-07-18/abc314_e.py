import sys
import numpy as np

def expected_cost(N, M, wheels):
    # Initialize the expected cost array
    E = np.zeros(M + 1)
    
    # Iterate over each point from 1 to M
    for points in range(1, M + 1):
        min_cost = float('inf')
        
        # Evaluate each wheel
        for cost, P, scores in wheels:
            expected_points = sum(scores) / P
            if expected_points > 0:
                # Calculate the expected cost for this wheel
                expected_cost = cost + (M - points) / expected_points * E[max(0, points)]
                min_cost = min(min_cost, expected_cost)
        
        E[points] = min_cost
    
    return E[M]

def main():
    input_data = sys.stdin.read().strip().splitlines()
    first_line = input_data[0].split()
    N = int(first_line[0])
    M = int(first_line[1])
    
    wheels = []
    
    for i in range(1, N + 1):
        line = list(map(int, input_data[i].split()))
        C = line[0]
        P = line[1]
        S = line[2:2 + P]
        wheels.append((C, P, S))
    
    result = expected_cost(N, M, wheels)
    print(f"{result:.10f}")

if __name__ == "__main__":
    main()