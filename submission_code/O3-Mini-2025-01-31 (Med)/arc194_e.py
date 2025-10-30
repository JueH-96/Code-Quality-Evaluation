def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    X = int(next(it))
    Y = int(next(it))
    S = next(it).strip()
    T = next(it).strip()

    # First necessary condition: the total number of ones must be the same.
    if S.count('1') != T.count('1'):
        sys.stdout.write("No")
        return

    # Observation:
    # Operation A requires a segment of X zeros followed by Y ones and changes it into Y ones then X zeros.
    # Operation B does the reverse.
    # In each operation, a block of Y ones is shifted by exactly X positions (either left or right).
    # In particular, note that the ones coming from the affected block move by a multiple of X.
    # Thus if we label each position by its index (using 0-indexing) and look at the remainder modulo X,
    # every operation leaves the remainder of a position containing a 1 unchanged.
    #
    # Therefore, for it to be possible to transform S into T, for every r in 0, 1, ..., X-1,
    # the number of ones in S whose index (0-indexed) is r must equal
    # the number of ones in T whose index is r.
    
    cntS = [0] * X
    cntT = [0] * X
    
    for i, ch in enumerate(S):
        if ch == '1':
            cntS[i % X] += 1
    for i, ch in enumerate(T):
        if ch == '1':
            cntT[i % X] += 1
    
    if cntS == cntT:
        sys.stdout.write("Yes")
    else:
        sys.stdout.write("No")

if __name__ == '__main__':
    main()