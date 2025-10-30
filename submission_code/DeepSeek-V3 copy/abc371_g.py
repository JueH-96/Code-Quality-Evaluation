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
    
    # For each cycle, find the lex smallest A
    for cycle in cycles:
        # Get the A values in the cycle
        a_values = [A[i-1] for i in cycle]
        # Find the rotation that makes the sequence lex smallest
        min_a = min(a_values)
        min_index = a_values.index(min_a)
        # Rotate the cycle to start with min_a
        rotated_cycle = cycle[min_index:] + cycle[:min_index]
        # Assign the sorted a_values to the rotated cycle
        sorted_a = sorted(a_values)
        for i in range(len(rotated_cycle)):
            A[rotated_cycle[i]-1] = sorted_a[i]
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()