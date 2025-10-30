import sys
import math
from collections import deque

def main():
    N, D = map(int, sys.stdin.readline().split())
    coords = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    infected = [False] * N
    infected[0] = True
    queue = deque()
    queue.append(0)
    
    while queue:
        current = queue.popleft()
        x1, y1 = coords[current]
        for i in range(N):
            if not infected[i]:
                x2, y2 = coords[i]
                distance_sq = (x1 - x2)**2 + (y1 - y2)**2
                if distance_sq <= D**2:
                    infected[i] = True
                    queue.append(i)
    
    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()