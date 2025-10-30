def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    events = []
    total = 0
    for _ in range(n):
        a = int(next(it)); b = int(next(it))
        total += b
        # at day a+1 this medicine stops
        events.append((a + 1, b))
    # If already at day 1 the total is <= K, answer is 1
    if total <= K:
        print(1)
        return
    # Sort events by day
    events.sort()
    idx = 0
    N = len(events)
    # Process events in order; at each distinct day d we subtract
    # all b_i for which a_i+1 == d, and check if total <= K
    while idx < N:
        d = events[idx][0]
        sumb = 0
        # accumulate all events at this same day d
        while idx < N and events[idx][0] == d:
            sumb += events[idx][1]
            idx += 1
        total -= sumb
        if total <= K:
            print(d)
            return

# call main as required
main()