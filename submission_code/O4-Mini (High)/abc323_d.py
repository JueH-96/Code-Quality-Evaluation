import sys
def main():
    import sys
    data = sys.stdin.readline
    from heapq import heapify, heappop, heappush

    n = int(data())
    counts = {}
    heap = []
    for _ in range(n):
        s, cnt = map(int, data().split())
        counts[s] = cnt
        heap.append(s)
    heapify(heap)

    get = counts.get
    pop = heappop
    push = heappush

    # Process sizes in ascending order, merging pairs and propagating.
    while heap:
        x = pop(heap)
        cntx = get(x, 0)
        if cntx < 2:
            continue
        # We have at least one pair to merge.
        pairs = cntx >> 1
        # Keep the remainder (0 or 1) at size x.
        if cntx & 1:
            counts[x] = 1
        else:
            # If no remainder, remove to save memory.
            del counts[x]
        # Propagate merged slimes to size 2*x.
        y = x << 1
        old = get(y, 0)
        new = old + pairs
        counts[y] = new
        # Only schedule y if it can merge further.
        if new >= 2:
            push(heap, y)

    # The remaining counts are all 0 or 1; sum gives the final number of slimes.
    print(sum(counts.values()))

if __name__ == "__main__":
    main()