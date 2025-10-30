def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    M = int(data[0])
    S = data[1:4]

    best = None

    # Try each digit from '0' to '9' as the candidate stop symbol.
    for d in "0123456789":
        # For each reel, compute the times at which it displays digit d.
        # When stopping a reel at time t, it shows S_i[(t mod M)].
        # For each occurrence index j where S_i[j] == d, the valid stopping times are:
        # j, j+M, j+2*M, ... We only need a few extra cycles so that we can resolve
        # potential conflicts (since we require three distinct times overall).
        cand = []  # List for each reel of candidate stop times.
        valid = True
        for i in range(3):
            options = []
            for j, ch in enumerate(S[i]):
                if ch == d:
                    # We'll consider a few cycles (k = 0,1,2) which is enough for M up to 100.
                    for k in range(3):
                        options.append(j + k * M)
            if not options:
                valid = False
                break  # If one reel never shows digit d, skip this candidate.
            options.sort()
            cand.append(options)
        if not valid:
            continue

        # Now check all combinations of candidate times
        # ensuring that the three selected times (one per reel) are distinct.
        for t0 in cand[0]:
            for t1 in cand[1]:
                if t1 == t0:
                    continue
                for t2 in cand[2]:
                    if t2 == t0 or t2 == t1:
                        continue
                    T = max(t0, t1, t2)
                    if best is None or T < best:
                        best = T

    if best is None:
        print(-1)
    else:
        print(best)

if __name__ == '__main__':
    main()