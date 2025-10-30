import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    P = int(data[3]) - 1  # Convert to 0-based indexing
    Q = int(data[4]) - 1  # Convert to 0-based indexing
    grid = []
    index = 5
    for _ in range(H):
        row = list(map(int, data[index:index + W]))
        grid.append(row)
        index += W
    
    # Initial strength
    current_strength = grid[P][Q]
    # Mark start position as absorbed
    absorbed = [[False] * W for _ in range(H)]
    absorbed[P][Q] = True
    # Priority queue: (-S_absorbed, r, c)
    pq = []
    # Directions for adjacent cells: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Function to add adjacent slimes to pq
    def add_adjacent(r, c):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not absorbed[nr][nc]:
                S_absorbed = grid[nr][nc]
                if S_absorbed * X < current_strength:
                    heapq.heappush(pq, (-S_absorbed, nr, nc))
    
    # Initial addition of adjacent slimes to starting position
    add_adjacent(P, Q)
    
    # Absorption loop
    while pq:
        # Pop the slime with the largest S_absorbed
        neg_S, r, c = heapq.heappop(pq)
        S_absorbed = -neg_S
        if S_absorbed * X < current_strength:
            # Absorb the slime
            current_strength += S_absorbed
            absorbed[r][c] = True
            # Add its adjacent slimes to pq
            add_adjacent(r, c)
        else:
            continue
    
    # Output the final strength
    print(current_strength)

if __name__ == '__main__':
    main()