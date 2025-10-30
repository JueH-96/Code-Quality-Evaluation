from collections import defaultdict

# Read input
N, M = map(int, input().split())
photos = [list(map(int, input().split())) for _ in range(M)]

# Create a dictionary to store the positions of each person in each photo
positions = defaultdict(list)
for i in range(M):
    for j, person in enumerate(photos[i]):
        positions[person].append((i, j))

# Count the number of pairs of people who did not stand next to each other in any photo
count = 0
for i in range(1, N+1):
    for j in range(i+1, N+1):
        # Check if the two people did not stand next to each other in any photo
        if all(abs(p1[1] - p2[1]) > 1 for p1 in positions[i] for p2 in positions[j]):
            count += 1

print(count)