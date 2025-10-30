from collections import deque

def main():
    N, M = map(int, input().split())
    grid = [list(input().strip()) for _ in range(N)]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize visited and touched grids
    visited = [[False for _ in range(M)] for _ in range(N)]
    touched = [[False for _ in range(M)] for _ in range(N)]
    
    # Start at (2,2) which is (1,1) in 0-based index
    start_i, start_j = 1, 1
    queue = deque()
    queue.append((start_i, start_j))
    visited[start_i][start_j] = True
    touched[start_i][start_j] = True
    
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            ni, nj = i, j
            # Move in the direction until hitting a rock
            while True:
                ni += di
                nj += dj
                if ni < 0 or ni >= N or nj < 0 or nj >= M or grid[ni][nj] == '#':
                    ni -= di
                    nj -= dj
                    break
            if not visited[ni][nj]:
                visited[ni][nj] = True
                queue.append((ni, nj))
            # Mark all squares passed through as touched
            ni, nj = i, j
            while True:
                ni += di
                nj += dj
                if ni < 0 or ni >= N or nj < 0 or nj >= M or grid[ni][nj] == '#':
                    break
                touched[ni][nj] = True
    
    # Count all touched ice squares
    count = 0
    for i in range(N):
        for j in range(M):
            if touched[i][j]:
                count += 1
    print(count)

if __name__ == "__main__":
    main()