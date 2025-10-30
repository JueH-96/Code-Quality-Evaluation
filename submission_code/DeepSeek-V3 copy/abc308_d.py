from collections import deque

def main():
    H, W = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Define the snuke sequence
    snuke = ['s', 'n', 'u', 'k', 'e']
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize visited array
    # visited[i][j][k] where k is the step in the snuke sequence
    visited = [[[False for _ in range(5)] for _ in range(W)] for _ in range(H)]
    
    # Start at (0, 0) with step 0
    start_char = grid[0][0]
    if start_char != snuke[0]:
        print("No")
        return
    
    q = deque()
    q.append((0, 0, 0))  # (i, j, step)
    visited[0][0][0] = True
    
    while q:
        i, j, step = q.popleft()
        if i == H-1 and j == W-1:
            print("Yes")
            return
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W:
                next_step = (step + 1) % 5
                expected_char = snuke[next_step]
                if grid[ni][nj] == expected_char and not visited[ni][nj][next_step]:
                    visited[ni][nj][next_step] = True
                    q.append((ni, nj, next_step))
    
    print("No")

if __name__ == "__main__":
    main()