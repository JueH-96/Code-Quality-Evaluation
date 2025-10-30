def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def can_transform(a, b, k):
        if len(a) != len(b):
            return False
        n = len(a)
        
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= k:
                    adj[i].append(j)

        for start_node in range(n):
            
            temp_a = a[:]
            
            q = [start_node]
            visited = [False] * n
            visited[start_node] = True
            
            while q:
                u = q.pop(0)
                
                for v in adj[u]:
                    if not visited[v]:
                        q.append(v)
                        visited[v] = True
            
            reachable_indices = [i for i in range(n) if visited[i]]
            
            
            groups_a = {}
            groups_b = {}
            
            for i in reachable_indices:
                val = a[i]
                if val not in groups_a:
                    groups_a[val] = 0
                groups_a[val] += 1
                
                val = b[i]
                if val not in groups_b:
                    groups_b[val] = 0
                groups_b[val] += 1
            
            
            if groups_a != groups_b:
                return False
        
        return True

    if can_transform(a, b, k):
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()