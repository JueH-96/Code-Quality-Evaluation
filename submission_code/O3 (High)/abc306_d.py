import sys

def main() -> None:
    # Read whole input at once and turn into an iterator of ints
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    n = data[0]
    it = iter(data[1:])

    NEG_INF = -10**30        # smaller than any achievable total value

    dp_healthy = 0           # best total when the stomach is healthy
    dp_upset   = NEG_INF     # best total when the stomach is upset

    for _ in range(n):
        x = next(it)         # 0 → antidotal, 1 → poisonous
        y = next(it)         # tastiness

        if x == 0:           # antidotal course
            # To end this step healthy:
            #   - skip while healthy        -> dp_healthy
            #   - eat while healthy         -> dp_healthy + y
            #   - eat while upset           -> dp_upset   + y
            new_healthy = dp_healthy
            eat_from_healthy = dp_healthy + y
            if eat_from_healthy > new_healthy:
                new_healthy = eat_from_healthy
            if dp_upset != NEG_INF:         # dp_upset might be unreachable
                eat_from_upset = dp_upset + y
                if eat_from_upset > new_healthy:
                    new_healthy = eat_from_upset

            # To end this step upset:
            #   - the only way is to already be upset and skip
            new_upset = dp_upset

        else:                # poisonous course
            # To end this step healthy:
            #   - must already be healthy and skip
            new_healthy = dp_healthy

            # To end this step upset:
            #   - already upset and skip          -> dp_upset
            #   - be healthy and eat this course  -> dp_healthy + y
            new_upset = dp_upset
            eat_from_healthy = dp_healthy + y
            if eat_from_healthy > new_upset:
                new_upset = eat_from_healthy

        dp_healthy, dp_upset = new_healthy, new_upset

    answer = max(0, dp_healthy, dp_upset)    # he may choose to eat nothing
    print(answer)

if __name__ == "__main__":
    main()