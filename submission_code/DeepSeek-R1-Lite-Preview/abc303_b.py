def main():
    import sys
    # Read N and M
    first_line = sys.stdin.readline().strip()
    while first_line == '':
        first_line = sys.stdin.readline().strip()
    N_M = first_line.split()
    N = int(N_M[0])
    M = int(N_M[1])
    
    # Read M lines of N integers each
    photos = []
    for _ in range(M):
        line = sys.stdin.readline().strip()
        while line == '':
            line = sys.stdin.readline().strip()
        photo = list(map(int, line.split()))
        photos.append(photo)
    
    # Set to store adjacent pairs
    adjacent_pairs = set()
    
    # Iterate through each photo and record adjacent pairs
    for photo in photos:
        for i in range(N - 1):
            pair = frozenset((photo[i], photo[i + 1]))
            adjacent_pairs.add(pair)
    
    # Calculate total number of pairs
    total_pairs = N * (N - 1) // 2
    
    # Calculate bad mood pairs
    bad_mood_pairs = total_pairs - len(adjacent_pairs)
    
    # Print the result
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()