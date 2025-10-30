def main():
    import sys, heapq
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    m = int(next(it))
    
    # events: list of (T, W, S)
    events = []
    for _ in range(m):
        T = int(next(it))
        W = int(next(it))
        S = int(next(it))
        events.append((T, W, S))
    # Although events are sorted by T in input we can assume they are in order.
    
    # The people initially in line are 1, 2, ..., n.
    # The front of the line will always be the person with the smallest original index.
    # We'll use a min-heap to represent the line.
    line = list(range(1, n + 1))
    heapq.heapify(line)
    
    # This is the heap for reentry events.
    # Each reentry event is a tuple (time, person), meaning the person returns at "time".
    reentries = []
    
    # We'll accumulate the amount of noodles each person gets.
    # Using 1-indexed list for convenience.
    ans = [0] * (n + 1)
    
    # Process each noodle drop event.
    for T, W, S in events:
        # Before processing the drop at time T, process all reentry events that occur on or before T.
        while reentries and reentries[0][0] <= T:
            rt, person = heapq.heappop(reentries)
            heapq.heappush(line, person)
        
        # At time T, if there is someone in the line, the person at the front gets the noodles.
        if line:
            person = heapq.heappop(line)
            ans[person] += W
            # Schedule the reentry of this person at time T + S.
            heapq.heappush(reentries, (T + S, person))
    
    # Print the total noodles received for each person from 1 to n.
    out_lines = "
".join(str(ans[i]) for i in range(1, n + 1))
    sys.stdout.write(out_lines)

if __name__ == '__main__':
    main()