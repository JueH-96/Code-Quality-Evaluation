def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))
    
    adjacent_pairs_set = set()
    for photo in photos:
        for j in range(n - 1):
            u = photo[j]
            v = photo[j+1]
            pair = tuple(sorted((u, v)))
            adjacent_pairs_set.add(pair)
            
    bad_mood_pair_count = 0
    for x in range(1, n + 1):
        for y in range(x + 1, n + 1):
            pair = tuple(sorted((x, y)))
            if pair not in adjacent_pairs_set:
                bad_mood_pair_count += 1
                
    print(bad_mood_pair_count)

if __name__ == '__main__':
    solve()