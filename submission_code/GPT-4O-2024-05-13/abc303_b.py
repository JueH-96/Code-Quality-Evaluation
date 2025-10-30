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
        photos.append([int(data[index + i]) for i in range(N)])
        index += N
    
    # Create a set to store pairs that are adjacent in any photo
    adjacent_pairs = set()
    
    for photo in photos:
        for i in range(N - 1):
            a, b = photo[i], photo[i + 1]
            if a > b:
                a, b = b, a
            adjacent_pairs.add((a, b))
    
    # Count all possible pairs
    all_pairs = set()
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            all_pairs.add((i, j))
    
    # Bad mood pairs are those that are not in adjacent_pairs
    bad_mood_pairs = all_pairs - adjacent_pairs
    
    print(len(bad_mood_pairs))

if __name__ == "__main__":
    main()