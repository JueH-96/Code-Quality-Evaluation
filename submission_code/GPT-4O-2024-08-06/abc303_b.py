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
        photo = list(map(int, data[index:index + N]))
        photos.append(photo)
        index += N
    
    # Set to store pairs that have been adjacent in any photo
    adjacent_pairs = set()
    
    for photo in photos:
        for j in range(N - 1):
            # Add both (a, b) and (b, a) to handle unordered pairs
            a, b = photo[j], photo[j + 1]
            if a > b:
                a, b = b, a
            adjacent_pairs.add((a, b))
    
    # Count pairs that may be in a bad mood
    bad_mood_count = 0
    
    for x in range(1, N + 1):
        for y in range(x + 1, N + 1):
            if (x, y) not in adjacent_pairs:
                bad_mood_count += 1
    
    print(bad_mood_count)

main()