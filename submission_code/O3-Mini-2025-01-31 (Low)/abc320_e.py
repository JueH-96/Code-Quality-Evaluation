def main():
    import sys
    from collections import deque
    import heapq
    input = sys.stdin.readline

    # Read input
    N, M = map(int, input().split())
    # list of noodle events, each as (T, W, S)
    events = [tuple(map(int, input().split())) for _ in range(M)]

    # result array: total noodle each person gets
    result = [0] * N

    # queue of people present in the row - initial order is 1,...,N
    queue = deque(range(1, N+1))

    # min-heap for scheduled return events: each event is (time, person)
    returns = []
    
    # Process each noodle event in order (T_i strictly increasing)
    for T, W, S in events:
        # Process all return events that occur at or before T
        while returns and returns[0][0] <= T:
            ret_time, person = heapq.heappop(returns)
            queue.append(person)
        # At time T, if there is anyone in the queue, the front gets the noodles.
        if queue:
            person = queue.popleft()
            result[person-1] += W
            # schedule the return for this person at time T+S
            heapq.heappush(returns, (T+S, person))
        # Else, if the queue is empty, nothing happens.
    
    # There is no requirement to process returns after M noodle events,
    # because no more noodle events occur.
    # Print result for each person.
    sys.stdout.write("
".join(map(str, result)))
    
if __name__ == '__main__':
    main()