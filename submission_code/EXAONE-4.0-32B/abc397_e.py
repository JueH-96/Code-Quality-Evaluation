import sys
sys.setrecursionlimit(300000)

def main():
    data = sys.stdin.read().split()
    if not data: 
        print("No")
        return
    it = iter(data)
    n = int(next(it))
    K = int(next(it))
    total_vertices = n * K
    adj = [[] for _ in range(total_vertices + 1)]
    
    for _ in range(total_vertices - 1):
        u = int(next(it))
        v = int(next(it))
        adj[u].append(v)
        adj[v].append(u)
        
    if K == 1:
        print("Yes")
        return
        
    def dfs(u, par):
        pending = []
        for v in adj[u]:
            if v == par:
                continue
            res = dfs(v, u)
            if res == -1:
                return -1
            if res != 0:
                pending.append(res)
                
        if len(pending) == 0:
            return 1
        elif len(pending) == 1:
            a = pending[0]
            if a + 1 > K:
                return -1
            elif a + 1 == K:
                return 0
            else:
                return a + 1
        else:
            if len(pending) != 2:
                return -1
            a, b = pending
            if a + b == K - 1:
                return 0
            else:
                return -1
                
    result = dfs(1, -1)
    print("Yes" if result == 0 else "No")

if __name__ == "__main__":
    main()