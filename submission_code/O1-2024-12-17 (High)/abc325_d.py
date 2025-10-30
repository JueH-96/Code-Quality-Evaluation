def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    intervals = []
    p = 1
    for _ in range(N):
        T = int(input_data[p])
        D = int(input_data[p+1])
        intervals.append((T, T + D))  # (start, end)
        p += 2

    # Sort by start time
    intervals.sort(key=lambda x: x[0])

    import heapq
    heap = []  # min-heap for end times
    count = 0
    t = 0      # current time at which we attempt to print
    i = 0      # index to scan intervals by start time

    # Process until we've considered all intervals and used up what we can
    while i < N or heap:
        # If no active intervals, jump time forward to the next interval's start
        if not heap:
            t = max(t, intervals[i][0])

        # Add all intervals that have started by time t
        while i < N and intervals[i][0] <= t:
            heapq.heappush(heap, intervals[i][1])  # push the end time
            i += 1

        if heap:
            # Take the interval with the smallest end time
            earliest_end = heapq.heappop(heap)
            # If it can accommodate printing at time t
            if earliest_end >= t:
                count += 1
                t += 1  # must wait 1 microsecond before next print
            # If earliest_end < t, we simply discard it and try again
        else:
            # No interval can be printed now; move to next interval's start (if any)
            if i < N:
                t = intervals[i][0]

    print(count)

# Call main() as required
if __name__ == "__main__":
    main()