def main():
    import sys
    import heapq

    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    chords = []
    for _ in range(N):
        x = int(next(it))
        y = int(next(it))
        # For each chord, order the endpoints so that a < b.
        if x < y:
            chords.append((x, y))
        else:
            chords.append((y, x))
    
    # For chords drawn on a circle with vertices in convex position, 
    # it is a well‐known fact that two chords (a, b) and (c, d) (with a < c)
    # intersect if and only if a < c < b < d.
    # We can therefore “sweep” the circle in the increasing order of point numbers
    # and use a data structure to hold those chords that have started (at a)
    # but not yet finished (at b). When a chord j starts (at a_j), every chord i in the active set 
    # satisfies a_i < a_j. Thus, if there is any active chord with end b_i < b_j then we have:
    # a_i < a_j < b_i < b_j, meaning chords i and j intersect.
    
    # Generate events for sweep: For each chord (a,b) we let:
    #   (a, 0, b) be the start event and (b, 1, b) be the end event.
    # (We use type 0 for start so that if ever endpoints coincided they would be processed before an end; 
    #  however, all endpoints are distinct in our problem.)
    events = []
    for a, b in chords:
        events.append((a, 0, b))
        events.append((b, 1, b))
    events.sort(key=lambda x: x[0])
    
    # Active chords will be maintained in a min-heap with their end value (b).
    active = []
    # We also maintain a dictionary for lazy removals from the heap.
    removed = {}

    # Clean the heap top from values that have been logically removed.
    def clean_heap():
        while active and removed.get(active[0], 0) > 0:
            cur = heapq.heappop(active)
            removed[cur] -= 1
            if removed[cur] == 0:
                del removed[cur]
    
    # Process events in increasing order.
    for pos, typ, b_val in events:
        if typ == 0:  # start event for a chord with end b_val
            clean_heap()
            # Before adding the current chord j (with end b_val), all active chords i have a_i < pos (which is a_j).
            # If any active chord i has b_i (its end) less than b_val, then
            # a_i < a_j < b_i < b_val holds and the chords intersect.
            if active and active[0] < b_val:
                sys.stdout.write("Yes")
                return
            heapq.heappush(active, b_val)
        else:  # end event: remove the chord ending at b_val from active set.
            removed[b_val] = removed.get(b_val, 0) + 1

    sys.stdout.write("No")

if __name__ == '__main__':
    main()