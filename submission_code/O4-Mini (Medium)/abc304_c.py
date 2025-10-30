import sys
from collections import deque

def main():
    input = sys.stdin.readline
    N, D = map(int, input().split())
    coords = [tuple(map(int, input().split())) for _ in range(N)]
    D2 = D * D

    visited = [False] * N
    visited[0] = True
    queue = deque([0])

    while queue:
        u = queue.popleft()
        ux, uy = coords[u]
        for v in range(N):
            if not visited[v]:
                vx, vy = coords[v]
                dx = ux - vx
                dy = uy - vy
                if dx*dx + dy*dy <= D2:
                    visited[v] = True
                    queue.append(v)

    out = sys.stdout
    for v in visited:
        out.write("Yes
" if v else "No
")

if __name__ == "__main__":
    main()