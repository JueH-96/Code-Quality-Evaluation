import sys
import heapq

input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
X = int(data[2])
P = int(data[3]) - 1  # converting to 0-based index
Q = int(data[4]) - 1  # converting to 0-based index

strengths = []
index = 5
for i in range(H):
    row = list(map(int, data[index:index+W]))
    strengths.append(row)
    index += W

# Directions for moving in the grid (up, down, left, right)
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Priority queue for the max-heap functionality, starting with Takahashi's initial position and strength
pq = [(-strengths[P][Q], P, Q)]
# Set to keep track of visited cells
visited = set()
visited.add((P, Q))

max_strength = strengths[P][Q]

while pq:
    current_strength, r, c = heapq.heappop(pq)
    current_strength = -current_strength  # Convert back from negative used for max-heap

    # Check all 4 possible directions
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
            neighbor_strength = strengths[nr][nc]
            if neighbor_strength < current_strength / X:
                # Takahashi can absorb this slime
                new_strength = current_strength + neighbor_strength
                max_strength = max(max_strength, new_strength)
                visited.add((nr, nc))
                heapq.heappush(pq, (-new_strength, nr, nc))

print(max_strength)