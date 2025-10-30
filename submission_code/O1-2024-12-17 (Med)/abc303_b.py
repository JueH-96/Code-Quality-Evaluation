def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    
    # Read the M photos
    photos = []
    for _ in range(M):
        row = list(map(int, data[idx:idx+N]))
        photos.append(row)
        idx += N

    # Create a 2D matrix to track adjacency (who stood next to whom)
    adjacent = [[False] * N for _ in range(N)]

    # Mark adjacency for each pair that stood next to each other in any photo
    for photo in photos:
        for i in range(N-1):
            p1 = photo[i] - 1  # converting 1-based to 0-based
            p2 = photo[i+1] - 1
            adjacent[p1][p2] = True
            adjacent[p2][p1] = True

    # Count how many distinct pairs never stood next to each other
    bad_mood_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not adjacent[i][j]:
                bad_mood_pairs += 1

    print(bad_mood_pairs)

# Do not forget to call main().
if __name__ == "__main__":
    main()