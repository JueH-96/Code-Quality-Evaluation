def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    
    # Use a 2D boolean matrix (or set) to mark if a given pair was ever adjacent.
    # Since pairs are unordered, we store them as (min, max).
    adjacent = set()
    
    for _ in range(M):
        # Read one photo order, a list of N integers.
        photo = [int(next(it)) for _ in range(N)]
        for i in range(N-1):
            a, b = photo[i], photo[i+1]
            # store pair in sorted order.
            if a > b:
                a, b = b, a
            adjacent.add((a, b))
    
    total_pairs = N * (N - 1) // 2
    result = total_pairs - len(adjacent)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()