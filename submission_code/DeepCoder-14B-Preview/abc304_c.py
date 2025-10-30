import sys
from collections import deque

def main():
    n, d = map(int, sys.stdin.readline().split())
    points = [tuple(map(int, sys.stdin.readline().split())) for _ in range(n)]
    
    infected = [False] * n
    infected[0] = True  # Person 1 is at index 0
    queue = deque([0])
    
    d_squared = d * d
    
    while queue:
        i = queue.popleft()
        for j in range(n):
            if i == j:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            if dx * dx + dy * dy <= d_squared and not infected[j]:
                infected[j] = True
                queue.append(j)
    
    for status in infected:
        print("Yes" if status else "No")

if __name__ == "__main__":
    main()