def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    k = int(next(it))
    # Read the permutation and convert it to 0-indexed.
    perm = [int(next(it)) for _ in range(n)]
    f = [p - 1 for p in perm]

    # The operation updates f into f^2 each time.
    # After K operations, the resulting permutation f_final = f^(2^K).
    # We achieve this by decomposing f into cycles.
    
    ans = [None] * n  # This will hold f_final(i) for each i, in 0-indexed form.
    visited = [False] * n

    for i in range(n):
        if not visited[i]:
            # Build the cycle starting at i.
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = f[cur]
            L = len(cycle)
            # We need to move each element forward by 2^K positions in its cycle.
            # For each cycle of length L, compute shift = 2^K mod L
            shift = pow(2, k, L)
            for idx, pos in enumerate(cycle):
                # f^(2^K) on element 'pos' is at cycle[(idx + shift) mod L]
                ans[pos] = cycle[(idx + shift) % L]

    # Convert the results back to 1-indexed and output.
    sys.stdout.write(" ".join(str(x + 1) for x in ans))

if __name__ == '__main__':
    main()