from collections import deque
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    D_sq = D * D
    coords = []
    for i in range(N):
        x = int(data[2 + 2*i])
        y = int(data[3 + 2*i])
        coords.append( (x, y) )
    infected = [False] * N
    queue = deque()
    infected[0] = True
    queue.append(0)
    while queue:
        current = queue.popleft()
        x1, y1 = coords[current]
        for i in range(N):
            if not infected[i]:
                x2, y2 = coords[i]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx*dx + dy*dy
                if dist_sq <= D_sq:
                    infected[i] = True
                    queue.append(i)
    for status in infected:
        print("Yes" if status else "No")

if __name__ == "__main__":
    main()