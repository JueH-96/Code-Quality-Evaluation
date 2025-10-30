def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return

    n = int(input_data[0])
    events = []
    index = 1
    for i in range(n):
        l = int(input_data[index])
        r = int(input_data[index+1])
        index += 2
        # Mark the start of the interval with type 0 and the end with type 1.
        # Since intervals are closed, if a start and an end have the same coordinate,
        # we want to count them as intersecting so we process the start first.
        events.append((l, 0))
        events.append((r, 1))
    
    # Sort events by coordinate, using type so that start events (0) come before end events (1)
    events.sort(key=lambda x: (x[0], x[1]))

    active_intervals = 0
    intersections = 0

    # Sweep through events:
    # For each new interval (start event), it intersects with all intervals that are currently active.
    for point, typ in events:
        if typ == 0:
            intersections += active_intervals
            active_intervals += 1
        else:
            active_intervals -= 1
    
    sys.stdout.write(str(intersections))
    
if __name__ == '__main__':
    main()