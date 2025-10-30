# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))
    
    # Create a mapping from index to its position in the cycle
    visited = [False] * (N + 1)
    cycles = []
    
    for i in range(1, N+1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current-1]
            cycles.append(cycle)
    
    # For each cycle, find the minimal rotation
    for cycle in cycles:
        # Create a list of A values in the cycle
        A_cycle = [A[i-1] for i in cycle]
        # Find the minimal rotation
        min_rotation = 0
        min_A = A_cycle
        for i in range(1, len(cycle)):
            rotated = A_cycle[i:] + A_cycle[:i]
            if rotated < min_A:
                min_A = rotated
                min_rotation = i
        # Apply the minimal rotation
        rotated_A = A_cycle[min_rotation:] + A_cycle[:min_rotation]
        for idx, val in zip(cycle, rotated_A):
            A[idx-1] = val
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()