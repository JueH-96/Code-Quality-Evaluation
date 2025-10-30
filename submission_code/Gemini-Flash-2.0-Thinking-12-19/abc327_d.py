import collections

def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))
    
    for i in range(m):
        if a[i] == b[i]:
            print("No")
            return
            
    adj = collections.defaultdict(list)
    for i in range(m):
        u, v = a[i], b[i]
        adj[u].append(v)
        adj[v].append(u)
        
    color = [-1] * (n + 1)
    
    for start_node in range(1, n + 1):
        if color[start_node] == -1:
            queue = collections.deque([start_node])
            color[start_node] = 0
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        queue.append(v)
                    elif color[v] == color[u]:
                        print("No")
                        return
                        
    print("Yes")

if __name__ == '__main__':
    solve()