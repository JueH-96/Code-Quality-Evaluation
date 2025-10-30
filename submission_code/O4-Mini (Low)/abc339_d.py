from collections import deque
import sys

def main():
    input = sys.stdin.readline
    N = int(input().strip())
    grid = [input().rstrip() for _ in range(N)]
    
    # Locate the two players and mark empty cells
    players = []
    empty = [[False]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if grid[i][j] != '#':
                empty[i][j] = True
            if grid[i][j] == 'P':
                players.append((i, j))
    
    (r1, c1), (r2, c2) = players
    
    # Flatten position (r, c) into index idx = r*N + c
    M = N * N
    start1 = r1 * N + c1
    start2 = r2 * N + c2
    
    # Precompute, for each index and each of 4 directions, the resulting index
    # If moving is blocked or out of range, stay in place.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    next_pos = [[0]*M for _ in range(4)]
    
    for idx in range(M):
        rr = idx // N
        cc = idx % N
        for d, (dr, dc) in enumerate(dirs):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < N and 0 <= nc < N and empty[nr][nc]:
                next_pos[d][idx] = nr * N + nc
            else:
                next_pos[d][idx] = idx
    
    # BFS over pairs of positions
    # visited will be a bytearray of length M*M
    total_states = M * M
    visited = bytearray(total_states)
    
    dq = deque()
    start_key = start1 * M + start2
    visited[start_key] = 1
    dq.append((start1, start2, 0))
    
    while dq:
        p1, p2, dist = dq.popleft()
        # Try each direction
        for d in range(4):
            np1 = next_pos[d][p1]
            np2 = next_pos[d][p2]
            if np1 == np2:
                # They meet
                print(dist + 1)
                return
            key = np1 * M + np2
            if not visited[key]:
                visited[key] = 1
                dq.append((np1, np2, dist + 1))
    
    # If BFS exhausts, impossible
    print(-1)

if __name__ == "__main__":
    main()