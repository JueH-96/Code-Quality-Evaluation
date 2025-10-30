import heapq

def main():
    h, w = map(int, input().split())
    grid = [list(input().strip()) for _ in range(h)]
    
    start = None
    goal = None
    for i in range(h):
        for j in range(w):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'T':
                goal = (i, j)
    
    n = int(input())
    medicines = {}
    for _ in range(n):
        r, c, e = map(int, input().split())
        r -= 1  # convert to 0-based index
        c -= 1
        medicines[(r, c)] = e
    
    # Directions: up, down, left, right
    directions = [ (-1, 0), (1, 0), (0, -1), (0, 1) ]
    
    # Priority queue: (negative energy, i, j, used_medicines)
    # used_medicines is a frozenset of tuples (i,j) where the medicine was used
    heap = []
    start_used = frozenset()
    heapq.heappush(heap, (0, start[0], start[1], start_used))
    
    visited = set()
    
    while heap:
        neg_e, i, j, used = heapq.heappop(heap)
        current_energy = -neg_e
        
        # Check if we've reached the goal
        if (i, j) == goal:
            print("Yes")
            return
        
        # Create the state key
        state = (i, j, current_energy, used)
        if state in visited:
            continue
        visited.add(state)
        
        # Move to adjacent cells
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                if grid[ni][nj] != '#' and current_energy > 0:
                    new_energy = current_energy - 1
                    new_used = used  # used_medicines remains the same
                    new_state = (ni, nj, new_energy, new_used)
                    if new_state not in visited:
                        # Push with negative energy to simulate max-heap
                        heapq.heappush(heap, (-new_energy, ni, nj, new_used))
        
        # Use the medicine in the current cell if available and not used
        if (i, j) in medicines:
            e = medicines[(i, j)]
            # Create a new used set
            new_used_set = set(used)
            new_used_set.add((i, j))
            new_used_frozen = frozenset(new_used_set)
            new_energy = e
            new_state = (i, j, new_energy, new_used_frozen)
            if new_state not in visited:
                heapq.heappush(heap, (-new_energy, i, j, new_used_frozen))
    
    # If we exhausted all possibilities and didn't reach the goal
    print("No")

if __name__ == "__main__":
    main()