from collections import deque
import sys
input = sys.stdin.readline

def main():
    H, W = map(int, input().split())
    S = [input().strip() for _ in range(H)]
    ans = 0
    seen = [[False] * W for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if S[i][j] == "#" and not seen[i][j]:
                ans += 1
                q = deque([(i, j)])
                while q:
                    x, y = q.popleft()
                    seen[x][y] = True
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < H and 0 <= ny < W and S[nx][ny] == "#" and not seen[nx][ny]:
                                q.append((nx, ny))
    print(ans)

main()