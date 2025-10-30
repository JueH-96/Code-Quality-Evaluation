def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    # Create a list to store the events.
    # Each interval [l, r] will create two events:
    # (l, 0) representing the start event and (r, 1) representing the end event.
    # Using 0 for start ensures that if an interval starts exactly when another ends,
    # we count them as intersecting, as required.
    events = [None] * (2 * n)
    idx = 1
    pos = 0
    for _ in range(n):
        l = int(data[idx])
        r = int(data[idx + 1])
        idx += 2
        events[pos] = (l, 0)  # start event
        events[pos + 1] = (r, 1)  # end event
        pos += 2

    # Sort events by coordinate; in the case of a tie, the start event (0) comes before the end event (1)
    events.sort(key=lambda x: (x[0], x[1]))

    active = 0  # number of intervals currently active (started but not ended)
    ans = 0    # count of intersecting pairs

    # Process each event in increasing order
    for point, typ in events:
        if typ == 0:  # Start event
            ans += active  # Each active interval intersects with the new interval
            active += 1
        else:  # End event
            active -= 1

    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()