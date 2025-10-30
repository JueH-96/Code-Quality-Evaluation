import collections
import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    index = 2
    graph = [[] for _ in range(n+1)]
    
    for i in range(m):
        a = int(data[index]); b = int(data[index+1]); x = int(data[index+2]); y = int(data[index+3])
        index += 4
        graph[a].append((b, x, y))
        graph[b].append((a, -x, -y))
    
    coords = [None] * (n+1)
    q = collections.deque()
    coords[1] = (0, 0)
    q.append(1)
    
    while q:
        u = q.popleft()
        for v, dx, dy in graph[u]:
            if coords[v] is None:
                new_x = coords[u][0] + dx
                new_y = coords[u][1] + dy
                coords[v] = (new_x, new_y)
                q.append(v)
    
    for i in range(1, n+1):
        if coords[i] is None:
            print("undecidable")
        else:
            print(f"{coords[i][0]} {coords[i][1]}")

if __name__ == "__main__":
    main()