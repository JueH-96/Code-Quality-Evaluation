MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx]); idx += 1
    M = int(data[idx]); idx += 1
    K = int(data[idx]); idx += 1

    edges = []
    # Add cycle edges
    for u in range(1, N + 1):
        if u == N:
            v = 1
        else:
            v = u + 1
        edges.append((u, v))
    # Add M edges
    for _ in range(M):
        x = int(data[idx]); idx += 1
        y = int(data[idx]); idx += 1
        edges.append((x, y))
    
    # Initialize DP arrays
    current = [0] * (N + 1)
    current[1] = 1  # Starting at vertex 1

    next_step = [0] * (N + 1)

    for _ in range(K):
        # Reset next_step to zero for each step
        for i in range(N + 1):
            next_step[i] = 0
        # Process each edge
        for u, v in edges:
            next_step[v] = (next_step[v] + current[u]) % MOD
        # Prepare for next iteration
        current, next_step = next_step, current  # Swap references

    # Sum all possible end vertices
    total = sum(current[1:N+1]) % MOD
    print(total)

if __name__ == '__main__':
    main()