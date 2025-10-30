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
    
    # Initialize a set to store pairs that are adjacent in at least one photo
    adjacent_pairs = set()
    
    for photo in photos:
        for i in range(N-1):
            x = photo[i]
            y = photo[i+1]
            if x < y:
                adjacent_pairs.add((x, y))
            else:
                adjacent_pairs.add((y, x))
    
    # Total possible pairs is C(N, 2)
    total_pairs = N * (N - 1) // 2
    
    # The number of pairs that are not adjacent in any photo is total_pairs - len(adjacent_pairs)
    bad_mood_pairs = total_pairs - len(adjacent_pairs)
    
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()