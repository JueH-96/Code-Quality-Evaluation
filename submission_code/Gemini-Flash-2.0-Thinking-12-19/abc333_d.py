import collections

def solve():
    n = int(input())
    if n == 2:
        input()
        print(1)
        return
    edges = []
    for _ in range(n - 1):
        u, v = map(int, input().split())
        edges.append((u, v))
    
    adjacency_list = collections.defaultdict(list)
    for u, v in edges:
        adjacency_list[u].append(v)
        adjacency_list[v].append(u)
        
    neighbors_of_1 = adjacency_list[1]
    if not neighbors_of_1:
        print(1)
        return
    if len(neighbors_of_1) <= 1:
        print(1)
        return
        
    subtree_sizes = []
    for neighbor in neighbors_of_1:
        visited = {1}
        queue = collections.deque([neighbor])
        visited.add(neighbor)
        count = 0
        while queue:
            u = queue.popleft()
            count += 1
            for v in adjacency_list[u]:
                if v not in visited and v != 1:
                    visited.add(v)
                    queue.append(v)
        subtree_sizes.append(count)
        
    sum_sizes = sum(subtree_sizes)
    max_size = max(subtree_sizes) if subtree_sizes else 0
    
    print(1 + sum_sizes - max_size)

if __name__ == '__main__':
    solve()