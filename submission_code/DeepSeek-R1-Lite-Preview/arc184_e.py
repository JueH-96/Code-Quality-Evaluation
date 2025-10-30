def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    N, M = map(int, sys.stdin.readline().split())
    sequences = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Precompute O^x for x from 0 to M*2 (arbitrary upper limit)
    # But to optimize, we'll find periods
    from collections import defaultdict
    orbits = {}
    visited = {}
    for seq in sequences:
        if seq in visited:
            continue
        current = seq
        orbit = []
        while current not in orbits:
            orbits[current] = []
            orbit.append(current)
            # Compute O(current)
            next_seq = tuple(sum(current[:k+1]) % 2 for k in range(M))
            orbits[current].append(next_seq)
            current = next_seq
        # Assign period
        visited[seq] = len(orbit)
    
    # Now, for each pair, find f(i,j)
    # But this is too slow for N=1e6
    # Instead, group sequences by their orbit representatives
    # And compute the sum based on the periods
    pass  # Placeholder, actual implementation needs optimization

if __name__ == '__main__':
    main()