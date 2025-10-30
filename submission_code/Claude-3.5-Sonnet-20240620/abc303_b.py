# YOUR CODE HERE
def are_adjacent(a, b, photo):
    for i in range(len(photo) - 1):
        if (photo[i] == a and photo[i+1] == b) or (photo[i] == b and photo[i+1] == a):
            return True
    return False

N, M = map(int, input().split())
photos = [list(map(int, input().split())) for _ in range(M)]

bad_mood_pairs = 0

for i in range(1, N):
    for j in range(i + 1, N + 1):
        if all(not are_adjacent(i, j, photo) for photo in photos):
            bad_mood_pairs += 1

print(bad_mood_pairs)