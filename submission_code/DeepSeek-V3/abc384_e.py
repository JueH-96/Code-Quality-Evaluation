import sys
import heapq

def main():
    H, W, X = map(int, sys.stdin.readline().split())
    P, Q = map(int, sys.stdin.readline().split())
    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(H)]
    
    # Adjust to 0-based index
    P -= 1
    Q -= 1
    
    # Initialize Takahashi's strength
    takahashi_strength = grid[P][Q]
    
    # Create a priority queue (min-heap) for adjacent slimes
    # We will store (strength, i, j)
    heap = []
    
    # Directions for adjacent cells
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize the heap with adjacent slimes
    for dx, dy in directions:
        x = P + dx
        y = Q + dy
        if 0 <= x < H and 0 <= y < W:
            heapq.heappush(heap, (grid[x][y], x, y))
    
    # Process the heap
    while heap:
        s, i, j = heapq.heappop(heap)
        if s < takahashi_strength / X:
            takahashi_strength += s
            # Add new adjacent slimes
            for dx, dy in directions:
                x = i + dx
                y = j + dy
                if 0 <= x < H and 0 <= y < W and (x != P or y != Q):
                    heapq.heappush(heap, (grid[x][y], x, y))
        else:
            # Since the heap is a min-heap, no further slimes can be absorbed
            break
    
    print(takahashi_strength)

if __name__ == "__main__":
    main()