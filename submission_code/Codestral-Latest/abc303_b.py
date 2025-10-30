import sys
from itertools import combinations

# Read input
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

# Create a set to store pairs that are adjacent in any photo
adjacent_pairs = set()

for photo in photos:
    for i in range(N - 1):
        pair = (min(photo[i], photo[i + 1]), max(photo[i], photo[i + 1]))
        adjacent_pairs.add(pair)

# Calculate all possible pairs
all_pairs = set(combinations(range(1, N + 1), 2))

# Find pairs that are not adjacent in any photo
bad_mood_pairs = all_pairs - adjacent_pairs

# Output the number of bad mood pairs
print(len(bad_mood_pairs))