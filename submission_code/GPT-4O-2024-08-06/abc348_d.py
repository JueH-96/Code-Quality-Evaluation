# YOUR CODE HERE
import sys
import heapq

def solve():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    
    grid = []
    index = 2
    start = None
    goal = None
    
    for i in range(H):
        row = data[index]
        grid.append(row)
        for j in range(W):
            if row[j] == 'S':
                start = (i, j)
            elif row[j] == 'T':
                goal = (i, j)
        index += 1
    
    N = int(data[index])
    index += 1
    
    medicines = {}
    for _ in range(N):
        R = int(data[index]) - 1
        C = int(data[index + 1]) - 1
        E = int(data[index + 2])
        medicines[(R, C)] = E
        index += 3
    
    # Directions for moving in the grid
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Priority queue for BFS
    pq = []
    heapq.heappush(pq, (0, start[0], start[1]))  # (negative energy, row, col)
    
    # Visited dictionary to store the maximum energy we have seen at each cell
    visited = {}
    visited[(start[0], start[1])] = 0
    
    while pq:
        neg_energy, r, c = heapq.heappop(pq)
        energy = -neg_energy
        
        if (r, c) == goal:
            print("Yes")
            return
        
        # Try to move to adjacent cells
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                if energy > 0:
                    if (nr, nc) not in visited or visited[(nr, nc)] < energy - 1:
                        visited[(nr, nc)] = energy - 1
                        heapq.heappush(pq, (-(energy - 1), nr, nc))
        
        # Try to use medicine if available
        if (r, c) in medicines:
            new_energy = medicines[(r, c)]
            if (r, c) not in visited or visited[(r, c)] < new_energy:
                visited[(r, c)] = new_energy
                heapq.heappush(pq, (-new_energy, r, c))
    
    print("No")