N, M = map(int, input().split())
photos = []
for _ in range(M):
    photos.append(list(map(int, input().split())))

# For each pair of people, check if they were ever adjacent
adjacent = [[False] * (N+1) for _ in range(N+1)]

for photo in photos:
    for i in range(N-1):
        # Mark these two people as having stood next to each other
        p1, p2 = photo[i], photo[i+1]
        adjacent[p1][p2] = True
        adjacent[p2][p1] = True

# Count pairs that were never adjacent
bad_mood_pairs = 0
for i in range(1, N+1):
    for j in range(i+1, N+1):
        if not adjacent[i][j]:
            bad_mood_pairs += 1

print(bad_mood_pairs)