import sys
import threading

def main():
    import sys
    import heapq

    input = sys.stdin.readline
    N, M = map(int, input().split())
    supplies = [tuple(map(int, input().split())) for _ in range(M)]
    # supplies are already in increasing T order by problem statement

    # Initialize the queue of present people as a min-heap of their indices
    queue = list(range(1, N+1))
    heapq.heapify(queue)

    # Min-heap of pending returns: (return_time, person_index)
    returns = []

    # Answer array, 1-based indexing
    ans = [0] * (N + 1)

    # Process each supply in chronological order
    for T, W, S in supplies:
        # First process all returns that happen at or before time T
        while returns and returns[0][0] <= T:
            ret_time, person = heapq.heappop(returns)
            heapq.heappush(queue, person)
        # Now process the supply at time T
        if queue:
            front = heapq.heappop(queue)
            ans[front] += W
            # Schedule this person to return at time T + S
            heapq.heappush(returns, (T + S, front))
        # If queue is empty, noodles are wasted

    # Output results
    out = sys.stdout
    for i in range(1, N+1):
        out.write(str(ans[i]) + "
")

if __name__ == "__main__":
    main()