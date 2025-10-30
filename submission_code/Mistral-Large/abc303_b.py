import sys
from collections import defaultdict

def count_bad_mood_pairs(N, M, photos):
    # Create a dictionary to store neighbors for each person
    neighbors = defaultdict(set)

    # Populate the neighbors dictionary
    for photo in photos:
        for i in range(N - 1):
            neighbors[photo[i]].add(photo[i + 1])
            neighbors[photo[i + 1]].add(photo[i])

    # Count the number of pairs who may be in a bad mood
    bad_mood_count = 0
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if j not in neighbors[i]:
                bad_mood_count += 1

    return bad_mood_count

def main():
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

    result = count_bad_mood_pairs(N, M, photos)
    print(result)

if __name__ == "__main__":
    main()