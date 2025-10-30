def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    k = int(next(it))
    # X is given in 1-indexed form. We'll convert it to 0-indexed for easier handling.
    X = [int(next(it)) - 1 for _ in range(n)]
    A = [int(next(it)) for _ in range(n)]
    
    # We are to perform K operations where after each operation,
    # the new sequence B is defined by B[i] = A[X[i]].
    # This means that after K operations, the element at index i becomes
    # A[f^K(i)] where f is the mapping f(i) = X[i].
    # We can precompute the cycles of the mapping f.
    
    # res will hold the final transformed sequence.
    res = [0] * n
    visited = [False] * n

    for i in range(n):
        if not visited[i]:
            # Identify the cycle starting from i.
            cycle = []
            cur = i
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = X[cur]
            L = len(cycle)
            # If i is in position j in the cycle, after K operations the element in i will come from position (j - K mod L) in the cycle.
            shift = k % L
            for j in range(L):
                res[cycle[j]] = A[cycle[(j - shift) % L]]
    
    print(" ".join(map(str, res)))
    
if __name__ == '__main__':
    main()