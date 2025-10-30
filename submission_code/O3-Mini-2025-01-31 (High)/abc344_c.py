def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    
    # Read input sequences
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = int(next(it))
    B = [int(next(it)) for _ in range(M)]
    L = int(next(it))
    C = [int(next(it)) for _ in range(L)]
    Q = int(next(it))
    Xs = [int(next(it)) for _ in range(Q)]
    
    # Precompute all possible sums of one element from A and one from B.
    # There are at most 100*100 = 10000 combinations.
    sumAB = set()
    for a in A:
        for b in B:
            sumAB.add(a + b)
    
    # Precompute all possible triple sums: one from A, one from B, and one from C.
    # Total iterations are |sumAB| * L, which is at most 10000*100 = 1e6.
    tripleSums = set()
    for s in sumAB:
        for c in C:
            tripleSums.add(s + c)
    
    # For each query X, check if it exists in the precomputed tripleSums.
    output = []
    for x in Xs:
        if x in tripleSums:
            output.append("Yes")
        else:
            output.append("No")
            
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()