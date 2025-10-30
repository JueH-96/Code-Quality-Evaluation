# YOUR CODE HERE

N, M = map(int, input().split())
photos = [list(map(int, input().split())) for _ in range(M)]

bad_pairs = set()

for i in range(M):
    for j in range(N - 1):
        person1 = photos[i][j]
        person2 = photos[i][j + 1]
        if (person2, person1) not in bad_pairs:
            bad_pairs.add((person1, person2))

print(len(bad_pairs))