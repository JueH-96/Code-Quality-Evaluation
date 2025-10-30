# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))
    
    # Convert P to 0-based index for easier manipulation
    P = [p - 1 for p in P]
    
    # To find the result after K operations, we need to find the cycle length
    visited = [False] * N
    cycles = []
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            cycles.append(cycle)
    
    # Now we know the cycles, we can compute the result after K operations
    result = [0] * N
    for cycle in cycles:
        cycle_len = len(cycle)
        k_mod = K % cycle_len
        for i in range(cycle_len):
            result[cycle[i]] = cycle[(i + k_mod) % cycle_len]
    
    # Convert result back to 1-based index
    result = [r + 1 for r in result]
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()