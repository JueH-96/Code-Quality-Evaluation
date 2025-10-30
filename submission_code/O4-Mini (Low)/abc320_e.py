import sys
import threading

def main():
    import sys
    data = sys.stdin
    n, m = map(int, data.readline().split())
    # Read noodle events
    events = [tuple(map(int, data.readline().split())) for _ in range(m)]
    # events are (T_i, W_i, S_i), and T_i strictly increasing

    import heapq

    # Min-heap of available people by original index (1..n)
    available = list(range(1, n+1))
    heapq.heapify(available)

    # Min-heap of return events: (return_time, person_index)
    returns = []

    # Answer array
    ans = [0] * (n + 1)

    # Process each noodle event in time order
    # Before each, process all return events up to that time (inclusive)
    for T, W, S in events:
        # Bring back all who return at or before T
        while returns and returns[0][0] <= T:
            ret_time, person = heapq.heappop(returns)
            heapq.heappush(available, person)
        # Give noodles to the front if any
        if available:
            person = heapq.heappop(available)
            ans[person] += W
            # Schedule return
            heapq.heappush(returns, (T + S, person))

    # Print results for persons 1..n
    out = sys.stdout
    for i in range(1, n+1):
        out.write(str(ans[i]) + "
")

if __name__ == "__main__":
    main()