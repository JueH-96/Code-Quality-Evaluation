def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    X = list(map(lambda x: int(x)-1, data[2:2+N]))  # zero-based permutation
    A = list(map(int, data[2+N:2+2*N]))
    
    # We'll find the final array B such that B[i] = A[p^K(i)].
    # Instead of fast exponentiation of permutation (which can be large in memory),
    # we use cycle decomposition. For each cycle c = [c0, c1, ..., c(L-1)],
    # we have B[c[i]] = A[c[(i + K) mod L]].
    
    visited = [False]*N
    B = [0]*N
    
    for start in range(N):
        if not visited[start]:
            # Find the cycle starting at 'start'.
            cycle = []
            cur = start
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = X[cur]
            
            L = len(cycle)
            shift = K % L  # we only need K modulo the cycle length
            
            # Fill the values in B according to the cycle shift.
            for i in range(L):
                B[cycle[i]] = A[cycle[(i + shift) % L]]
    
    print(" ".join(map(str, B)))

def main():
    solve()

# Call solve() explicitly (as requested).
solve()