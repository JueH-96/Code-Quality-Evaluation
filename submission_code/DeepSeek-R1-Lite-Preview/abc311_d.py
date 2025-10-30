from collections import deque

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N, M = map(int, data[0].split())
    grid = [list(data[i+1]) for i in range(N)]
    
    # Directions: up, down, left, right
    directions = [(-1,0), (1,0), (0,-1), (0,1)]
    
    # Initialize touched and enqueued matrices
    touched = [[False for _ in range(M)] for _ in range(N)]
    enqueued = [[False for _ in range(M)] for _ in range(N)]
    
    queue = deque()
    start_i, start_j = 1, 1  # 0-indexed
    queue.append((start_i, start_j))
    touched[start_i][start_j] = True
    enqueued[start_i][start_j] = True
    
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            ni, nj = i + di, j + dj
            # Slide until hitting a rock
            path = []
            while 0 <= ni < N and 0 <= nj < M and grid[ni][nj] == '.':
                if not touched[ni][nj]:
                    touched[ni][nj] = True
                path.append((ni, nj))
                ni += di
                nj += dj
            # Stopping position is the last ice square before the rock
            if path:
                stopping_i, stopping_j = path[-1]
                if not enqueued[stopping_i][stopping_j]:
                    enqueued[stopping_i][stopping_j] = True
                    queue.append((stopping_i, stopping_j))
    
    # Count the number of touched squares
    count = 0
    for i in range(N):
        for j in range(M):
            if grid[i][j] == '.' and touched[i][j]:
                count += 1
    print(count)

if __name__ == '__main__':
    main()