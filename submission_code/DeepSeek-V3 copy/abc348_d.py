from collections import deque

def main():
    H, W = map(int, input().split())
    grid = [list(input().strip()) for _ in range(H)]
    N = int(input())
    medicines = []
    for _ in range(N):
        R, C, E = map(int, input().split())
        medicines.append((R-1, C-1, E))  # converting to 0-based index
    
    # Find start and goal positions
    start = None
    goal = None
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    if not start or not goal:
        print("No")
        return
    
    # Create a dictionary to map positions to medicines
    med_dict = {}
    for idx, (r, c, e) in enumerate(medicines):
        med_dict[(r, c)] = e
    
    # Initialize BFS
    visited = {}  # key: (i,j), value: energy
    queue = deque()
    queue.append((start[0], start[1], 0))
    visited[(start[0], start[1])] = 0
    
    while queue:
        i, j, energy = queue.popleft()
        if (i, j) == goal:
            print("Yes")
            return
        # Check if there is a medicine at current cell
        if (i, j) in med_dict:
            new_energy = med_dict[(i, j)]
            if new_energy > energy:
                energy = new_energy
                visited[(i, j)] = energy
        # Explore neighbors
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] != '#':
                    if energy > 0:
                        new_energy = energy - 1
                        if (ni, nj) not in visited or new_energy > visited.get((ni, nj), -1):
                            visited[(ni, nj)] = new_energy
                            queue.append((ni, nj, new_energy))
    print("No")

if __name__ == "__main__":
    main()