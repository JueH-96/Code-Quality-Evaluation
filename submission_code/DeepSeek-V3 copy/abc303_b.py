# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    photos = []
    index = 2
    for _ in range(M):
        photo = list(map(int, data[index:index+N]))
        photos.append(photo)
        index += N
    
    adjacent = set()
    for photo in photos:
        for i in range(N-1):
            x = photo[i]
            y = photo[i+1]
            if x < y:
                adjacent.add((x, y))
            else:
                adjacent.add((y, x))
    
    total_pairs = N * (N - 1) // 2
    adjacent_pairs = len(adjacent)
    bad_mood_pairs = total_pairs - adjacent_pairs
    
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()