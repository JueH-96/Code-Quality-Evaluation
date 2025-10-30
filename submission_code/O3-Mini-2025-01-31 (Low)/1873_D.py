def main():
    import sys, bisect
    data = sys.stdin.read().split()
    t = int(data[0])
    pos = 1
    results = []
    for _ in range(t):
        n = int(data[pos])
        k = int(data[pos+1])
        pos += 2
        s = data[pos]
        pos += 1

        # Gather all indices with a black cell 'B'.
        black_positions = [i for i, ch in enumerate(s) if ch == 'B']
        
        # Greedy algorithm: cover as many consecutive black cells as possible with a single k-length interval.
        operations = 0
        i = 0
        m = len(black_positions)
        while i < m:
            # Starting from the leftmost uncovered black cell.
            current = black_positions[i]
            # Choosing an interval starting such that it covers from current to current+k-1.
            cover_end = current + k - 1
            # Using binary search to find the first black cell index that is not covered by this interval.
            i = bisect.bisect_right(black_positions, cover_end, i, m)
            operations += 1
        
        results.append(str(operations))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()