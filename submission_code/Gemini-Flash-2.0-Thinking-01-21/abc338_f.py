import collections
import heapq

def solve():
    n, m = map(int, input().split())
    edges = []
    for _ in range(m):
        u, v, w = map(int, input().split())
        edges.append(((u, v), w))
    
    adj = collections.defaultdict(list)
    for (u, v), w in edges:
        adj[u].append((v, w))
        
    def get_reachable_vertices(start_node):
        reachable = set()
        q = collections.deque([start_node])
        reachable.add(start_node)
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in reachable:
                    reachable.add(v)
                    q.append(v)
        return reachable
        
    possible_walk = False
    for i in range(1, n + 1):
        reachable_set = get_reachable_vertices(i)
        if len(reachable_set) == n:
            possible_walk = True
            break
            
    if not possible_walk:
        print("No")
        return
        
    dp = collections.defaultdict(lambda: float('inf'))
    pq = []
    
    for i in range(1, n + 1):
        mask = 1 << (i - 1)
        dp[(i, mask)] = 0
        heapq.heappush(pq, (0, i, mask))
        
    min_cost = float('inf')
    
    while pq:
        cost, u, mask = heapq.heappop(pq)
        if cost > dp[(u, mask)]:
            continue
            
        if mask == (1 << n) - 1:
            min_cost = min(min_cost, cost)
            
        for v, weight in adj[u]:
            next_mask = mask | (1 << (v - 1))
            new_cost = cost + weight
            if new_cost < dp[(v, next_mask)]:
                dp[(v, next_mask)] = new_cost
                heapq.heappush(pq, (new_cost, v, next_mask))
                
    if min_cost == float('inf'):
        print("No") # Should not happen if reachability is checked correctly
    else:
        print(min_cost)

if __name__ == '__main__':
    solve()