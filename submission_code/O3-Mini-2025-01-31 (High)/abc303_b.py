def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    
    # This set will store pairs that have been adjacent in at least one photo.
    adjacent_pairs = set()
    
    for _ in range(M):
        # Read one photo's permutation.
        photo = [int(next(it)) for _ in range(N)]
        for j in range(N - 1):
            # Get the pair of consecutive persons.
            u, v = photo[j], photo[j + 1]
            # Store in sorted order so that (1,2) and (2,1) are treated the same.
            if u > v:
                u, v = v, u
            adjacent_pairs.add((u, v))
    
    # Total pairs among N people.
    total_pairs = N * (N - 1) // 2
    # Pairs that may be in a bad mood are those never adjacent in any photo.
    result = total_pairs - len(adjacent_pairs)
    
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()