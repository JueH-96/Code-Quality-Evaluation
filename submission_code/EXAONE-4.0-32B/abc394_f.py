import sys
sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    graph = [[] for _ in range(n)]
    index = 1
    for i in range(n-1):
        a = int(data[index]); b = int(data[index+1]); index += 2
        a -= 1
        b -= 1
        graph[a].append(b)
        graph[b].append(a)
    
    parent = [-1] * n
    children = [[] for _ in range(n)]
    stack = [0]
    while stack:
        u = stack.pop()
        for v in graph[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            children[u].append(v)
            stack.append(v)
    
    order = []
    stack = [0]
    while stack:
        u = stack.pop()
        order.append(u)
        for v in children[u]:
            stack.append(v)
    order.reverse()
    
    INF = -10**18
    open1 = [0] * n
    open3 = [INF] * n
    closed = [INF] * n
    
    for u in order:
        open1[u] = 1
        
        child_closed_max = INF
        max1 = max2 = max3 = max4 = INF
        max_open3 = INF
        
        for v in children[u]:
            if closed[v] > child_closed_max:
                child_closed_max = closed[v]
                
            w_val = max(open1[v], open3[v])
            if w_val > max1:
                max4 = max3
                max3 = max2
                max2 = max1
                max1 = w_val
            elif w_val > max2:
                max4 = max3
                max3 = max2
                max2 = w_val
            elif w_val > max3:
                max4 = max3
                max3 = w_val
            elif w_val > max4:
                max4 = w_val
                
            if open3[v] > max_open3:
                max_open3 = open3[v]
                
        if max3 != INF:
            open3[u] = 1 + max1 + max2 + max3
            
        candidates = [child_closed_max]
        if max4 != INF:
            candidates.append(1 + max1 + max2 + max3 + max4)
        if max_open3 != INF:
            candidates.append(1 + max_open3)
            
        closed[u] = max(candidates)
        
    ans = closed[0]
    if ans < 5:
        print(-1)
    else:
        print(ans)

if __name__ == '__main__':
    main()