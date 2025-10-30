def main():
    import sys
    import heapq

    data = sys.stdin.buffer.read().split()
    if not data:
        return

    it = iter(data)
    # Read number of people and number of events
    N = int(next(it))
    M = int(next(it))

    # Read events data (T, W, S) for each event.
    events = []
    for _ in range(M):
        T = int(next(it))
        W = int(next(it))
        S = int(next(it))
        events.append((T, W, S))

    # active: a min-heap containing persons currently in the row.
    # The queue is always ordered by the original order (i.e., smaller person ID is ahead)
    active = list(range(1, N + 1))
    heapq.heapify(active)

    # ret_queue: min-heap of scheduled return events. Each item is a tuple (return_time, person)
    ret_queue = []

    # ans[i] will hold total noodles received by person i (using 1-indexing)
    ans = [0] * (N + 1)

    # Process each event in increasing order of time T.
    # At each event time T the following steps occur:
    # 1. Process all scheduled returns that have return time <= T.
    # 2. If there is anyone in the active queue, pop the front person (smallest id),
    #    give that person the noodles, and schedule their return at T + S.
    for T, W, S in events:
        # Process any persons returning at or before current time T.
        while ret_queue and ret_queue[0][0] <= T:
            ret_time, person = heapq.heappop(ret_queue)
            heapq.heappush(active, person)
        # If the row is not empty, the front person gets the noodles.
        if active:
            person = heapq.heappop(active)
            ans[person] += W
            # Schedule this person to return at time T + S.
            return_time = T + S
            heapq.heappush(ret_queue, (return_time, person))
        # If no one is in the row, we do nothing (no noodles are assigned).

    # Write final output: one line per person, in order from person 1 to person N.
    out_lines = []
    for i in range(1, N + 1):
        out_lines.append(str(ans[i]))
    sys.stdout.write("
".join(out_lines))


if __name__ == "__main__":
    main()