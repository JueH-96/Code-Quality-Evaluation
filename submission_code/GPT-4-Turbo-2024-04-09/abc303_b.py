def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    photos = []
    index = 2
    for _ in range(M):
        photos.append(list(map(int, data[index:index+N])))
        index += N
    
    # Set to store pairs that have been adjacent
    adjacent = set()
    
    for photo in photos:
        for i in range(N-1):
            for j in range(i+1, N):
                # Mark the pair as adjacent
                adjacent.add((photo[i], photo[j]))
                adjacent.add((photo[j], photo[i]))
    
    # Calculate the number of pairs that have never been adjacent
    bad_mood_count = 0
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            if (i, j) not in adjacent and (j, i) not in adjacent:
                bad_mood_count += 1
    
    print(bad_mood_count)

main()