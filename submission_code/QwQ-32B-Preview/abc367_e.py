import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:2+N]))
    A = list(map(int, data[2+N:2+2*N]))
    
    # Make X 0-based
    X = [x - 1 for x in X]
    
    # Find cycle decomposition
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = X[current]
            cycles.append(cycle)
    
    # Create mapping after K operations
    mapping = [0] * N
    for cycle in cycles:
        L = len(cycle)
        shift = K % L
        for j in range(L):
            original_pos = cycle[j]
            new_pos = cycle[(j + shift) % L]
            mapping[original_pos] = new_pos + 1  # convert to 1-based
    
    # Construct A'
    A_prime = [A[mapping[i] - 1] for i in range(N)]
    
    # Print the result
    print(' '.join(map(str, A_prime)))

if __name__ == '__main__':
    main()