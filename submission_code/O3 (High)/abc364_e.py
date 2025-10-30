import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, X, Y = map(int, input().split())
    dishes = [tuple(map(int, input().split())) for _ in range(N)]

    INF = Y + 1          # larger than any feasible salt value
    # dp[sweet][cnt] = minimal salt that is necessary to obtain exactly `cnt`
    # dishes whose total sweetness is `sweet`
    dp = [[INF] * (N + 1) for _ in range(X + 1)]
    dp[0][0] = 0

    # for every sweetness value we remember which counts are currently reachable
    reachable_counts = [set() for _ in range(X + 1)]
    reachable_counts[0].add(0)

    active_sweets = {0}          # all sweetness sums that are reachable now
    best_cnt = 0                 # best number of dishes that still keeps both totals within limits

    for used, (a, b) in enumerate(dishes, 1):
        new_active = set(active_sweets)
        for s in list(active_sweets):               # iterate only over actually reachable sweet sums
            for c in list(reachable_counts[s]):     # and only over counts that exist for this sum
                salt_so_far = dp[s][c]
                ns = s + a
                if ns > X:            # sweetness would already break the limit
                    continue
                nsalt = salt_so_far + b
                if nsalt > Y:         # saltiness would already break the limit
                    continue

                # better (smaller) salt for the same (ns, c+1)?
                if nsalt < dp[ns][c + 1]:
                    dp[ns][c + 1] = nsalt
                    reachable_counts[ns].add(c + 1)
                    new_active.add(ns)
                    if c + 1 > best_cnt:
                        best_cnt = c + 1
        active_sweets = new_active

    #  Snuke can always try to eat one more dish – the very next one will make him stop – so
    #  the final answer is at most best_cnt + 1 (but never larger than N).
    if best_cnt == N:
        print(N)
    else:
        print(best_cnt + 1)


if __name__ == "__main__":
    main()