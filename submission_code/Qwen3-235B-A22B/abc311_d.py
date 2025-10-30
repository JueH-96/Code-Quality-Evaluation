import sys
from collections import deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = [sys.stdin.readline().strip() for _ in range(N)]
    
    visited = [[False] * M for _ in range(N)]
    processed = [[False] * M for _ in range(N)]
    queue = deque()
    
    start_r, start_c = 1, 1
    processed[start_r][start_c] = True
    queue.append((start_r, start_c))
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        
        for dr, dc in directions:
            path = []
            current_r, current_c = r, c
            path.append((current_r, current_c))
            
            while True:
                next_r = current_r + dr
                next_c = current_c + dc
                if 0 <= next_r < N and 0 <= next_c < M:
                    if S[next_r][next_c] == '.':
                        current_r = next_r
                        current_c = next_c
                        path.append((current_r, current_c))
                    else:
                        break
                else:
                    break
            
            for x, y in path:
                visited[x][y] = True
            
            stop_r, stop_c = current_r, current_c
            if not processed[stop_r][stop_c]:
                processed[stop_r][stop_c] = True
                queue.append((stop_r, stop_c))
    
    count = sum(sum(row) for row in visited)
    print(count)

if __name__ == "__main__":
    main()