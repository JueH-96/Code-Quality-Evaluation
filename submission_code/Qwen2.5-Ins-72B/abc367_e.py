# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:2+N]))
    A = list(map(int, data[2+N:]))
    
    # Adjust X to be 0-indexed
    X = [x - 1 for x in X]
    
    # Find the cycle
    visited = [False] * N
    cycle = []
    for i in range(N):
        if not visited[i]:
            current = i
            path = []
            while not visited[current]:
                visited[current] = True
                path.append(current)
                current = X[current]
            if current in path:
                cycle_start = path.index(current)
                cycle.extend(path[cycle_start:])
    
    # Compute the effective number of operations
    cycle_length = len(cycle)
    if cycle_length > 0:
        K = K % cycle_length
    
    # Perform the operations
    for _ in range(K):
        A = [A[X[i]] for i in range(N)]
    
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()