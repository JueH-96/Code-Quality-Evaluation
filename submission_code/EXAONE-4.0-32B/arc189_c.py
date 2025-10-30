import collections
import sys

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    
    n = int(data[0])
    X = int(data[1]) - 1
    A = list(map(int, data[2:2+n]))
    B = list(map(int, data[2+n:2+2*n]))
    P = list(map(lambda x: int(x)-1, data[2+2*n:2+3*n]))
    Q = list(map(lambda x: int(x)-1, data[2+3*n:2+4*n]))
    
    rev_red = [[] for _ in range(n)]
    rev_blue = [[] for _ in range(n)]
    for i in range(n):
        j = P[i]
        rev_red[j].append(i)
        j2 = Q[i]
        rev_blue[j2].append(i)
        
    reachable_red = [False] * n
    queue_red = collections.deque()
    queue_red.append(X)
    reachable_red[X] = True
    while queue_red:
        u = queue_red.popleft()
        for v in rev_red[u]:
            if not reachable_red[v]:
                reachable_red[v] = True
                queue_red.append(v)
                
    reachable_blue = [False] * n
    queue_blue = collections.deque()
    queue_blue.append(X)
    reachable_blue[X] = True
    while queue_blue:
        u = queue_blue.popleft()
        for v in rev_blue[u]:
            if not reachable_blue[v]:
                reachable_blue[v] = True
                queue_blue.append(v)
                
    for i in range(n):
        if A[i] == 1 and i != X:
            if not reachable_red[i]:
                print(-1)
                return
        if B[i] == 1 and i != X:
            if not reachable_blue[i]:
                print(-1)
                return
                
    needed_red = [False] * n
    needed_blue = [False] * n
    
    for i in range(n):
        if A[i] == 1 and i != X:
            current = i
            path = []
            while current != X and not needed_red[current]:
                path.append(current)
                current = P[current]
            for node in path:
                needed_red[node] = True
                
    for i in range(n):
        if B[i] == 1 and i != X:
            current = i
            path = []
            while current != X and not needed_blue[current]:
                path.append(current)
                current = Q[current]
            for node in path:
                needed_blue[node] = True
                
    count = 0
    for i in range(n):
        if needed_red[i] or needed_blue[i]:
            count += 1
            
    print(count)

if __name__ == "__main__":
    main()