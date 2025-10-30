def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    bets = []
    for _ in range(N):
        Ci = int(next(it))
        A = set(int(next(it)) for _ in range(Ci))
        bets.append((Ci, A))
    X = int(next(it))
    # Find all people who bet on X
    candidates = []
    for idx, (Ci, A) in enumerate(bets, start=1):
        if X in A:
            candidates.append((Ci, idx))
    # If nobody bet on X, print 0 and exit
    if not candidates:
        print(0)
        return
    # Determine the minimum number of bets among those who bet on X
    min_bets = min(ci for ci, _ in candidates)
    # Collect all indices with that minimum
    result = [idx for ci, idx in candidates if ci == min_bets]
    result.sort()
    # Output
    print(len(result))
    print(*result)

# Call main to execute
main()