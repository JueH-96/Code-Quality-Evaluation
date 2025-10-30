import sys
import threading

def main():
    import sys
    import heapq

    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline

    H, W = map(int, input().split())
    grid = [input().rstrip('
') for _ in range(H)]
    N = int(input())
    # Potion energy map: 0 if none, else E_i
    potion = [[0]*W for _ in range(H)]
    for _ in range(N):
        r, c, e = map(int, input().split())
        potion[r-1][c-1] = e

    # Find start and target
    sr = sc = tr = tc = -1
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                sr, sc = i, j
            elif grid[i][j] == 'T':
                tr, tc = i, j

    # E_max[r][c]: max energy we've had upon reaching (r,c)
    # Initialize to -1 (unvisited)
    E_max = [[-1]*W for _ in range(H)]
    # Priority queue: max-heap by energy
    # Stored as (-energy, r, c)
    pq = []
    # Start at S with energy 0
    E_max[sr][sc] = 0
    heapq.heappush(pq, (0, sr, sc))  # energy=0 -> -0=0

    # Directions: up, down, left, right
    dirs = [(-1,0),(1,0),(0,-1),(0,1)]

    while pq:
        neg_e, r, c = heapq.heappop(pq)
        e = -neg_e
        # If this state is outdated, skip
        if e < E_max[r][c]:
            continue
        # If we've reached the target, answer Yes
        if r == tr and c == tc:
            print("Yes")
            return
        # Option to use potion here
        p_e = potion[r][c]
        if p_e > e:
            # Using potion sets energy to p_e
            if p_e > E_max[r][c]:
                E_max[r][c] = p_e
                heapq.heappush(pq, (-p_e, r, c))
        # Move to neighbors if we have at least 1 energy
        if e >= 1:
            ne = e - 1
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    if ne > E_max[nr][nc]:
                        E_max[nr][nc] = ne
                        heapq.heappush(pq, (-ne, nr, nc))

    # If we exhaust all states without reaching T, answer No
    print("No")

if __name__ == "__main__":
    main()