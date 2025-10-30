def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    
    intervals = []
    coords = []
    idx = 1
    for _ in range(N):
        l = int(input_data[idx]); r = int(input_data[idx+1])
        idx += 2
        intervals.append((l, r))
        coords.append(l)
        coords.append(r)
    
    # Coordinate compression
    coords = sorted(set(coords))
    # Dictionary to map original coordinate to compressed index (1-based)
    comp = {}
    for i, v in enumerate(coords):
        comp[v] = i+1  # 1-based index
    
    # Transform intervals to compressed
    compressed_intervals = []
    for l, r in intervals:
        compressed_intervals.append((comp[l], comp[r]))
    
    # Sort by left coordinate
    compressed_intervals.sort(key=lambda x: x[0])
    
    # Fenwick(BIT) setup
    size = len(coords)  # max compressed coordinate
    fenwicks = [0]*(size+1)
    
    def fenwicks_update(i, val):
        while i <= size:
            fenwicks[i] += val
            i += i & -i
    
    def fenwicks_sum(i):
        s = 0
        while i > 0:
            s += fenwicks[i]
            i -= i & -i
        return s
    
    # Count intersections
    ans = 0
    for L, R in compressed_intervals:
        # Number of intervals (already processed) that have r_j >= L
        # i.e. fenwicks_sum(size) - fenwicks_sum(L-1)
        ans += fenwicks_sum(size) - fenwicks_sum(L-1)
        # Mark this interval's right endpoint
        fenwicks_update(R, 1)
    
    print(ans)

# Do not forget to call main()
main()