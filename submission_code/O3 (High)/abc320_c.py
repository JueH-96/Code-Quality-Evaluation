import sys

def main() -> None:
    M = int(sys.stdin.readline().strip())
    reels = [sys.stdin.readline().strip() for _ in range(3)]

    INF = 10 ** 9
    best_overall = INF

    # try every digit that could be shown
    for d in '0123456789':
        time_lists = []
        for s in reels:
            times = set()
            for idx, ch in enumerate(s):
                if ch == d:
                    # pressing at t shows character at position (t mod M)
                    # examine the first three cycles : k = 0,1,2
                    for k in range(3):
                        times.add(idx + k * M)
            time_lists.append(sorted(times))

        # the digit must appear on every reel
        if any(len(lst) == 0 for lst in time_lists):
            continue

        l1, l2, l3 = time_lists
        current_best = best_overall    # no need to look for worse solutions

        # three-nested loops with pruning
        for t1 in l1:
            if t1 >= current_best:
                break
            for t2 in l2:
                if t2 == t1:
                    continue
                m12 = max(t1, t2)
                if m12 >= current_best:
                    break
                for t3 in l3:
                    if t3 == t1 or t3 == t2:
                        continue
                    maxi = max(m12, t3)
                    if maxi < current_best:
                        current_best = maxi
                        # l3 is sorted – any later t3 only increases maxi
                        break
            # if the smallest possible maxi equals t1, we cannot
            # improve by choosing larger t1, so break early
            if current_best == t1:
                break

        best_overall = min(best_overall, current_best)

    print(-1 if best_overall == INF else best_overall)

if __name__ == "__main__":
    main()