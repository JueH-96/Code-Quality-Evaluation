import sys
import threading

def main():
    import sys
    from collections import deque
    
    input = sys.stdin.readline
    N, D = map(int, input().split())
    coords = [tuple(map(int, input().split())) for _ in range(N)]
    D2 = D * D
    
    infected = [False] * N
    infected[0] = True  # person 1 is index 0
    queue = deque([0])
    
    while queue:
        i = queue.popleft()
        x_i, y_i = coords[i]
        for j in range(N):
            if not infected[j]:
                x_j, y_j = coords[j]
                dx = x_i - x_j
                dy = y_i - y_j
                if dx*dx + dy*dy <= D2:
                    infected[j] = True
                    queue.append(j)
    
    out = sys.stdout
    for inf in infected:
        out.write("Yes
" if inf else "No
")

if __name__ == "__main__":
    main()