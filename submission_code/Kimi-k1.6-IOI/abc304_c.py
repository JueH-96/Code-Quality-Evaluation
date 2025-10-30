from collections import deque
import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr +=1
    D = int(input[ptr])
    ptr +=1
    D_squared = D * D
    coords = []
    for _ in range(N):
        x = int(input[ptr])
        y = int(input[ptr+1])
        coords.append((x, y))
        ptr +=2
    
    infected = [False] * N
    infected[0] = True
    q = deque([0])
    
    while q:
        u = q.popleft()
        for v in range(N):
            if not infected[v]:
                dx = coords[u][0] - coords[v][0]
                dy = coords[u][1] - coords[v][1]
                if dx * dx + dy * dy <= D_squared:
                    infected[v] = True
                    q.append(v)
    
    for i in range(N):
        print("Yes" if infected[i] else "No")

if __name__ == "__main__":
    main()