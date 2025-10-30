# YOUR CODE HERE
import sys
import heapq

def main():
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    X = int(data[2])
    P = int(data[3]) - 1
    Q = int(data[4]) - 1
    
    S = []
    index = 5
    for i in range(H):
        row = []
        for j in range(W):
            row.append(int(data[index]))
            index += 1
        S.append(row)
    
    # Directions for moving in the grid (up, down, left, right)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Priority queue to manage the cells to be processed
    pq = []
    heapq.heappush(pq, (-S[P][Q], P, Q))  # Use negative strength for max-heap
    visited = set()
    visited.add((P, Q))
    
    max_strength = S[P][Q]
    
    while pq:
        current_strength, x, y = heapq.heappop(pq)
        current_strength = -current_strength  # Convert back to positive
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < H and 0 <= ny < W and (nx, ny) not in visited:
                if S[nx][ny] < current_strength / X:
                    visited.add((nx, ny))
                    new_strength = current_strength + S[nx][ny]
                    heapq.heappush(pq, (-new_strength, nx, ny))
                    max_strength = max(max_strength, new_strength)
    
    print(max_strength)

if __name__ == "__main__":
    main()