from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    grid = []
    index = 2
    for _ in range(N):
        grid.append(list(data[index]))
        index += 1

    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Initialize touched and visited matrices
    touched = [[False for _ in range(M)] for _ in range(N)]
    visited = [[False for _ in range(M)] for _ in range(N)]
    
    # Starting position (0-indexed)
    start_i = 1
    start_j = 1
    touched[start_i][start_j] = True
    visited[start_i][start_j] = True
    count = 1  # Starting square is ice and touched
    
    # BFS queue
    queue = deque()
    queue.append((start_i, start_j))
    
    while queue:
        i, j = queue.popleft()
        
        for di, dj in directions:
            ni, nj = i, j
            path = []
            while True:
                ti, tj = ni + di, nj + dj
                if 0 <= ti < N and 0 <= tj < M and grid[ti][tj] == '.':
                    ni, nj = ti, tj
                    path.append((ni, nj))
                else:
                    break
            # ni, nj is the stopping position
            # Mark all cells in path as touched
            for pi, pj in path:
                if not touched[pi][pj]:
                    touched[pi][pj] = True
                    count += 1
            # Enqueue the stopping position if not visited and it's ice
            if (ni, nj) != (i, j) and not visited[ni][nj]:
                visited[ni][nj] = True
                queue.append((ni, nj))
    
    print(count)

if __name__ == "__main__":
    main()