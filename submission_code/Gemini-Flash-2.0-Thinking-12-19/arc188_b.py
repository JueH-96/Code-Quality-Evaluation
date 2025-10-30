import collections

def solve():
    n, k = map(int, input().split())
    if n <= 1:
        print("Yes" if n == 1 else "No")
        return
    
    adjacency_list = collections.defaultdict(list)
    for i in range(n):
        reflection_alice = (n - i) % n
        reflection_bob = (2 * k - i) % n
        adjacency_list[i].append(reflection_alice)
        adjacency_list[i].append(reflection_bob)
        
    visited = [False] * n
    components_count = 0
    for i in range(n):
        if not visited[i]:
            components_count += 1
            queue = collections.deque([i])
            visited[i] = True
            while queue:
                u = queue.popleft()
                for v in adjacency_list[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        
    if components_count <= 2:
        print("Yes")
    else:
        print("No")

t = int(input())
for _ in range(t):
    solve()