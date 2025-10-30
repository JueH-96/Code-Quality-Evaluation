def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    P = list(map(int, data[2:]))

    # Convert P to 0-based for easier cycle handling
    p = [x - 1 for x in P]
    
    visited = [False] * N
    ans = [0] * N  # will hold the final permutation in 0-based form
    
    for i in range(N):
        if not visited[i]:
            # Explore the cycle starting from i
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = p[current]
            
            # cycle now contains all indices in this cycle
            length = len(cycle)
            shift = K % length  # only the remainder matters
            
            # For each index in the cycle, compute its final position
            for idx_pos, idx in enumerate(cycle):
                ans[idx] = cycle[(idx_pos + shift) % length]
                
    # Convert back to 1-based and print
    # ans[i] is the 0-based final image of i, so we need ans[i] + 1
    print(" ".join(str(x+1) for x in ans))


# Do not forget to call main()
if __name__ == "__main__":
    main()