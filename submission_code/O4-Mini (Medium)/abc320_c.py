def main():
    import sys
    sys.setrecursionlimit(10000)
    data = sys.stdin.read().split()
    M = int(data[0])
    S = data[1:4]

    INF = 10**18
    best = INF
    # We need at most 3*M seconds to get three distinct press times
    MAXT = 3 * M

    for d in map(str, range(10)):
        # collect for each reel the possible stopping times <= MAXT
        reels_times = []
        ok = True
        for i in range(3):
            si = S[i]
            # positions where S[i][pos]==d
            pos = [j for j, ch in enumerate(si) if ch == d]
            if not pos:
                ok = False
                break
            times = []
            for p in pos:
                t = p
                # generate p + k*M
                while t <= MAXT:
                    times.append(t)
                    t += M
            # unique & sorted
            times = sorted(set(times))
            reels_times.append(times)
        if not ok:
            continue

        # union of all times
        all_times = sorted(set(reels_times[0] + reels_times[1] + reels_times[2]))
        # map time -> index
        time_to_idx = {t: i for i, t in enumerate(all_times)}
        # adjacency list from reel -> list of time-indices
        nbr = [[] for _ in range(3)]
        for i in range(3):
            lst = reels_times[i]
            out = nbr[i]
            for t in lst:
                out.append(time_to_idx[t])
            # sort for consistency (not strictly needed)
            out.sort()

        # try each T from 0..MAXT, stopping early if we find matching size 3
        ptr = 0
        L = len(all_times)
        # match_to stores for each time-index which reel it is matched to, or -1
        match_to = [-1] * L

        def dfs(u, seen, cutoff):
            # try to match reel u
            for v in nbr[u]:
                if v >= cutoff:
                    # time > T
                    break
                if seen[v]:
                    continue
                seen[v] = True
                w = match_to[v]
                if w == -1 or dfs(w, seen, cutoff):
                    match_to[v] = u
                    return True
            return False

        found = False
        for T in range(MAXT + 1):
            # expand ptr so that all_times[:ptr] <= T
            while ptr < L and all_times[ptr] <= T:
                ptr += 1
            # try to find matching of size 3 with only times < ptr
            # reset matching
            for i in range(ptr):
                match_to[i] = -1
            matched = 0
            # for each reel, try to find augmenting path
            for u in range(3):
                seen = [False] * ptr
                if dfs(u, seen, ptr):
                    matched += 1
                else:
                    break
            if matched == 3:
                # we can stop all three by time T with distinct press times
                best = min(best, T)
                found = True
                break
        # if best == 0, can't get smaller, but we still need to check other digits?
        # actually 0 is minimal possible, but no harm continuing

    if best == INF:
        print(-1)
    else:
        print(best)

if __name__ == "__main__":
    main()