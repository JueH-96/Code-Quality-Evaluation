def main():
    import sys
    import bisect

    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Build the initial counts of slimes by size.
    counts = {}
    for _ in range(n):
        s = int(next(it))
        c = int(next(it))
        counts[s] = counts.get(s, 0) + c

    # We will process sizes in increasing order. Since synthesis always produces a slime of size 2*x,
    # any newly created size will be larger than x, so we can safely process in increasing order.
    keys = sorted(counts.keys())
    keys_set = set(keys)

    i = 0
    while i < len(keys):
        size = keys[i]
        cnt = counts.get(size, 0)
        # For each size, we can pair up floor(cnt/2) slimes. Each pair creates one new slime of size 2*size.
        pairs = cnt // 2
        # After pairing, remainder remains (0 or 1).
        counts[size] = cnt % 2
        
        if pairs:
            new_size = size * 2
            counts[new_size] = counts.get(new_size, 0) + pairs
            # If this new size is not in keys, insert it to be processed later.
            if new_size not in keys_set:
                bisect.insort(keys, new_size)
                keys_set.add(new_size)
        i += 1

    # The minimum possible number of slimes is just the total number of slimes left.
    total_slimes = sum(counts[size] for size in counts)
    sys.stdout.write(str(total_slimes))
    
if __name__ == '__main__':
    main()