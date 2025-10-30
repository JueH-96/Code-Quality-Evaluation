def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))

    ever_adjacent = set()
    for photo in photos:
        for i in range(n - 1):
            person1 = photo[i]
            person2 = photo[i+1]
            ever_adjacent.add(tuple(sorted((person1, person2))))

    bad_mood_count = 0
    for p1 in range(1, n + 1):
        for p2 in range(p1 + 1, n + 1):
            if tuple(sorted((p1, p2))) not in ever_adjacent:
                bad_mood_count += 1

    print(bad_mood_count)

solve()