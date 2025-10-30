def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    # parse input
    it = iter(input_data)
    n = int(next(it))
    P = [int(next(it))-1 for _ in range(n)]  # convert to 0-indexed
    A = [int(next(it)) for _ in range(n)]
    
    # We'll simulate the operation f: A -> B where B[i] = A[P[i]].
    # The orbit is f^k(A) for k=0,1,..., and eventually cycles.
    # We compare arrays lexicographically.
    
    # For lex comparison, we can use tuple conversion.
    best = A[:]  # best array found so far
    best_tuple = tuple(best)
    current = A[:]  # current state, starting with no operation.
    
    # simulate until we return back to the initial state.
    # We use a loop; note that cycle length might be huge in worst case,
    # but typical inputs are expected to have moderate cycle order.
    first_state = A[:]
    while True:
        # perform one simultaneous operation
        nxt = [0]*n
        for i in range(n):
            nxt[i] = current[P[i]]
        current = nxt
        # if we are back to the beginning, break.
        if current == first_state:
            break
        # update best if current lexicographically smaller
        cur_tuple = tuple(current)
        if cur_tuple < best_tuple:
            best_tuple = cur_tuple
            best = current[:]
    
    # print best
    sys.stdout.write(" ".join(map(str, best)))
    
if __name__ == '__main__':
    main()