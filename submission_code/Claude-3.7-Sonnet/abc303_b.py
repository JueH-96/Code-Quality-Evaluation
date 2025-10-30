def solve():
    # Read input
    N, M = map(int, input().split())
    photos = []
    for _ in range(M):
        photo = list(map(int, input().split()))
        photos.append(photo)

    # Initialize a set to store pairs that are adjacent in at least one photo
    adjacent_pairs = set()

    # Check for adjacent pairs in each photo
    for photo in photos:
        for i in range(N - 1):
            # Add the pair (smaller_number, larger_number) to avoid duplicates
            person1, person2 = min(photo[i], photo[i + 1]), max(photo[i], photo[i + 1])
            adjacent_pairs.add((person1, person2))

    # Count pairs that may be in a bad mood (not adjacent in any photo)
    bad_mood_pairs = 0
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i, j) not in adjacent_pairs:
                bad_mood_pairs += 1

    print(bad_mood_pairs)

if __name__ == "__main__":
    solve()