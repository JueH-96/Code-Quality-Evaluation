import sys
import heapq


def main() -> None:
    """
    Simulate the Flowing Noodles event.

    row_heap   : min-heap that keeps the people who are currently in the row.
                 Because the initial order is 1…N, the person at the front is
                 always the smallest index that is present, therefore a
                 min-heap works.
    return_heap: min-heap of (return_time, person).  
                 When somebody leaves the row we register the time they will
                 come back.  
                 Before handling a noodles drop that happens at time t we move
                 everybody whose return_time ≤ t from return_heap back to the
                 row_heap.

    present[i] : whether person i is currently in the row (needed to avoid
                 inserting the same person twice into row_heap).

    Each drop event is processed once, each leave/return is processed once:
    overall complexity O((N+M) log N), well inside the limits.
    """
    sys.setrecursionlimit(1 << 25)
    it = sys.stdin.readline

    N, M = map(int, it().split())

    # everyone is in the row at the beginning
    row_heap = list(range(1, N + 1))
    heapq.heapify(row_heap)

    present = [False] * (N + 1)
    for i in range(1, N + 1):
        present[i] = True

    return_heap: list[tuple[int, int]] = []       # (time, id)
    total = [0] * (N + 1)                         # noodles obtained

    for _ in range(M):
        t, w, s = map(int, it().split())

        # first, let everybody who should have returned by now come back
        while return_heap and return_heap[0][0] <= t:
            _, pid = heapq.heappop(return_heap)
            if not present[pid]:                  # should always be True
                heapq.heappush(row_heap, pid)
                present[pid] = True

        # noodles drop
        if row_heap:
            pid = heapq.heappop(row_heap)         # person at the front
            total[pid] += w
            present[pid] = False                  # now out of the row
            heapq.heappush(return_heap, (t + s, pid))

    # print answers
    sys.stdout.write('
'.join(str(total[i]) for i in range(1, N + 1)) + '
')


if __name__ == '__main__':
    main()