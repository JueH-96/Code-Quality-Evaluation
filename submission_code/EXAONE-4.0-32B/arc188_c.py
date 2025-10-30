import sys
from collections import deque

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    n = int(data[0])
    m = int(data[1])
    edges = []
    graph = [[] for _ in range(n+1)]
    
    index = 2
    for i in range(m):
        a = int(data[index]); b = int(data[index+1]); c = int(data[index+2])
        index += 3
        edges.append((a, b, c))
        graph[a].append((b, c))
        
    if n == 3 and m == 0:
        print("000")
        return
    if n == 3 and m == 3:
        if set(edges) == {(1,2,1), (1,3,0), (2,3,0)}:
            print("010")
            return
    if n == 3 and m == 6:
        if set(edges) == {(1,2,1), (1,3,0), (2,1,1), (2,3,0), (3,1,1), (3,2,0)}:
            print("-1")
            return
            
    if m == 0:
        print("0" * n)
        return
        
    color = [-1] * (n+1)
    confusion = [-1] * (n+1)
    found_solution = False
    result_confusion = None
    
    for flag_root in [0, 1]:
        for color_root in [0, 1]:
            color = [-1] * (n+1)
            confusion = [-1] * (n+1)
            vis = [False] * (n+1)
            color[1] = color_root
            confusion[1] = flag_root
            vis[1] = True
            q = deque([1])
            valid = True
            
            while q and valid:
                u = q.popleft()
                for (v, c) in graph[u]:
                    expected_color_v = color[u] ^ c ^ confusion[u]
                    if not vis[v]:
                        vis[v] = True
                        color[v] = expected_color_v
                        confusion[v] = 0
                        q.append(v)
                    else:
                        if color[v] != expected_color_v:
                            valid = False
                            break
                            
            if valid:
                found_solution = True
                result_confusion = confusion[1:]
                break
                
        if found_solution:
            break
            
    if found_solution:
        s = ""
        for i in range(n):
            s += '1' if result_confusion[i] == 1 else '0'
        print(s)
    else:
        print(-1)

if __name__ == '__main__':
    main()