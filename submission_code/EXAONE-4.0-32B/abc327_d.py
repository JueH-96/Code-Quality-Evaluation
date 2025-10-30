import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    if not data:
        print("No")
        return
    
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+m]))
    B = list(map(int, data[2+m:2+2*m]))
    
    for i in range(m):
        if A[i] == B[i]:
            print("No")
            return
            
    graph = [[] for _ in range(n+1)]
    for i in range(m):
        a = A[i]
        b = B[i]
        graph[a].append(b)
        graph[b].append(a)
        
    color = [-1] * (n+1)
    dq = deque()
    
    for i in range(1, n+1):
        if color[i] == -1:
            color[i] = 0
            dq.append(i)
            while dq:
                u = dq.popleft()
                for v in graph[u]:
                    if color[v] == -1:
                        color[v] = color[u] ^ 1
                        dq.append(v)
                    else:
                        if color[v] == color[u]:
                            print("No")
                            return
                            
    print("Yes")

if __name__ == "__main__":
    main()