def main():
    import sys
    sys.setrecursionlimit(10**7)
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    
    X = [int(x) - 1 for x in data[2:2+N]]  # convert to 0-based
    A = list(map(int, data[2+N:2+2*N]))
    
    # We want to apply the permutation p(i)=X[i] exactly K times,
    # so for each i, the final position is A[p^K(i)].
    # We'll use cycle decomposition: for each cycle of length L,
    # p^K(i) is i advanced by K (mod L) positions around that cycle.
    
    visited = [False]*N
    B = [0]*N  # final array
    
    for start in range(N):
        if visited[start]:
            continue
        
        # Follow the cycle from 'start'
        cycle = []
        cur = start
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = X[cur]
        
        # Now cycle contains all indices of this cycle
        L = len(cycle)
        if L == 1:
            # Single element cycle
            B[cycle[0]] = A[cycle[0]]
        else:
            # Fill B for each element in the cycle
            # If cycle = [c0, c1, ..., c_{L-1}],
            # then B[c_j] = A[c_{(j+K) mod L}].
            k_mod = K % L
            for j in range(L):
                B[cycle[j]] = A[cycle[(j + k_mod) % L]]
    
    print(' '.join(map(str, B)))

# Call main()
if __name__ == "__main__":
    main()