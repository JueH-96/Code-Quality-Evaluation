# YOUR CODE HERE
from itertools import product

N, M = map(int, input().split())
positions = [list(map(int, input().split())) for _ in range(M)]

bad_mood_pairs = 0

for i, j in product(range(1, N+1), repeat=2):
    if i == j:
        continue
    in_same_photo = any(i in positions[k] and j in positions[k] for k in range(M))
    if not in_same_photo:
        left_neighbours = [positions[k].index(i) for k in range(M) if i in positions[k]]
        right_neighbours = [positions[k].index(j) for k in range(M) if j in positions[k]]
        if not any(abs(left - right) == 1 for left in left_neighbours for right in right_neighbours):
            bad_mood_pairs += 1

print(bad_mood_pairs)