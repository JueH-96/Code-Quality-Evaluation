def main():
    import sys
    data = sys.stdin.read().strip().split()
    it = iter(data)
    N = int(next(it))
    # We'll store for each person i: (Ci, set_of_bets)
    bets = []
    for _ in range(N):
        Ci = int(next(it))
        Ai = set()
        for _ in range(Ci):
            Ai.add(int(next(it)))
        bets.append((Ci, Ai))
    X = int(next(it))
    # Find all persons who bet on X
    candidates = []
    for idx, (Ci, Ai) in enumerate(bets, start=1):
        if X in Ai:
            candidates.append((Ci, idx))
    # If nobody bet on X
    if not candidates:
        print(0)
        return
    # Find the minimum Ci among those who bet on X
    min_bets = min(ci for ci, _ in candidates)
    # Collect all person indices with Ci == min_bets
    answer = [idx for ci, idx in candidates if ci == min_bets]
    answer.sort()
    # Output
    print(len(answer))
    print(" ".join(map(str, answer)))

if __name__ == "__main__":
    main()