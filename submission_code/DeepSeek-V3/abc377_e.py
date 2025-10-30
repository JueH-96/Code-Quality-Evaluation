# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:2+N]))
    
    # Precompute the cycles
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
    
    # For each element, find its position in the cycle and compute the new position after K steps
    result = [0] * N
    for cycle in cycles:
        m = len(cycle)
        for idx, val in enumerate(cycle):
            new_idx = (idx + K) % m
            result[val-1] = cycle[new_idx]
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()