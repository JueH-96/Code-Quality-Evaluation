def solve():
    n, m = map(int, input().split())
    photos = []
    for _ in range(m):
        photos.append(list(map(int, input().split())))
    
    adjacent_pairs = set()
    for photo in photos:
        for j in range(n - 1):
            person1 = photo[j]
            person2 = photo[j+1]
            pair = tuple(sorted((person1, person2)))
            adjacent_pairs.add(pair)
            
    total_pairs_count = n * (n - 1) // 2
    adjacent_count = len(adjacent_pairs)
    
    print(total_pairs_count - adjacent_count)

if __name__ == '__main__':
    solve()