def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    
    # We will record every pair of persons that appear adjacent in at least one photo.
    adjacent_pairs = set()
    
    for _ in range(M):
        # Read one photo, which is a permutation of 1..N
        photo = [int(next(it)) for _ in range(N)]
        # For each adjacent pair in the photo, add the pair (in sorted order) to the set.
        for j in range(N - 1):
            a, b = photo[j], photo[j + 1]
            if a > b:
                a, b = b, a
            adjacent_pairs.add((a, b))
    
    # Total pairs among N persons = N choose 2
    total_pairs = N * (N - 1) // 2
    
    # Those pairs that never appear as adjacent in any photo could be in "bad mood"
    result = total_pairs - len(adjacent_pairs)
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()