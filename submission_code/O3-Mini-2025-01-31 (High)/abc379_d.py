def main():
    import sys, heapq
    data = sys.stdin.read().split()
    if not data:
        return
    q = int(data[0])
    # Global variable that represents the number of days waited so far.
    g = 0
    # We'll maintain a max‐heap using Python's heapq (which is a min–heap)
    # For each plant we want to compare the stored value x defined as:
    #   x = 0 - (global value at insertion)
    # Its effective height later is: x + g.
    # To simulate a max–heap based on x we store -x.
    # When inserting a new plant at current g, since x = -g,
    # we store value = -x = g.
    heap = []
    out_lines = []
    pos = 1
    for _ in range(q):
        typ = data[pos]
        pos += 1
        if typ == "1":
            # Plant a new flower pot.
            # New plant's relative height x = 0 - g.
            # We push -x into the heap, so push current g.
            heapq.heappush(heap, g)
        elif typ == "2":
            # Wait T days.
            T = int(data[pos])
            pos += 1
            g += T
        elif typ == "3":
            # Harvest all plants with effective height >= H.
            # A plant with stored value x has effective height = x + g.
            # Harvest condition: x + g >= H  <==>  x >= H - g.
            H = int(data[pos])
            pos += 1
            threshold = H - g
            cnt = 0
            # In the heap we stored -x.
            # The maximum x among all plants = -heap[0].
            # So while the maximum x meets x >= threshold, remove it.
            while heap and (-heap[0]) >= threshold:
                heapq.heappop(heap)
                cnt += 1
            out_lines.append(str(cnt))
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()