def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    # Read permutation and convert to 0-indexed.
    p = [int(next(it)) - 1 for _ in range(n)]
    
    # Our function f is defined by f(i) = p[i]
    # Notice that each operation transforms the permutation P to one where:
    #   new_P[i] = P_{P_i} = f(f(i))
    # Hence, after one operation, we have applied f twice.
    # Therefore after K operations, we apply f^(2^K) to each i.
    # Our goal is to compute f^(2^K)(i) for every i.
    # A good approach is to find the cycles in the permutation f.
    #
    # In a cycle of length L, for an element at position pos in the cycle,
    # f^m(i) will be the element at position (pos + m mod L).
    # Therefore, for f^(2^K)(i) we need to compute shift = 2^K mod L.
    
    ans = [None] * n
    visited = [False] * n
    
    for i in range(n):
        if visited[i]:
            continue
        cycle = []
        cur = i
        # Build the cycle starting from index i.
        while not visited[cur]:
            visited[cur] = True
            cycle.append(cur)
            cur = p[cur]
        L = len(cycle)
        # Compute the number of steps as shift = 2^K mod L.
        shift = pow(2, k, L)
        # For each index in the cycle, update its final position.
        for idx, pos in enumerate(cycle):
            ans[pos] = cycle[(idx + shift) % L]
    
    # Convert back from 0-indexed to 1-indexed for output.
    output = " ".join(str(x + 1) for x in ans)
    sys.stdout.write(output)

if __name__ == '__main__':
    main()