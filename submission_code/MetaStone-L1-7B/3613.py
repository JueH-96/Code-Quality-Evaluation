import sys
from collections import deque

def main():
    sys.setrecursionlimit(1 << 25)
    initial = int(sys.stdin.readline())
    n1, m1 = map(int, sys.stdin.readline().split())
    day1 = []
    for _ in range(m1):
        u, v, r = map(float, sys.stdin.readline().split())
        day1.append([u, v, r])
    n2, m2 = map(int, sys.stdin.readline().split())
    day2 = []
    for _ in range(m2):
        u, v, r = map(float, sys.stdin.readline().split())
        day2.append([u, v, r])
    
    def compute_max_product(edges, initial_node):
        max_product = {c: 0.0 for c in all_c}
        max_product[initial_node] = 1.0
        in_degree = {c: 0 for c in all_c}
        adj = {c: [] for c in all_c}
        for u, v, r in edges:
            adj[u].append(v)
            in_degree[v] += 1
        
        queue = deque()
        for c in all_c:
            if in_degree[c] == 0:
                queue.append(c)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if max_product[v] < max_product[u] * r:
                    max_product[v] = max_product[u] * r
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        return max_product
    
    all_c = set()
    for u in day1:
        all_c.add(u[0])
        all_c.add(u[1])
    for u in day2:
        all_c.add(u[0])
        all_c.add(u[1])
    all_c = list(all_c)
    
    m1 = compute_max_product(day1, initial)
    reversed_day2_graph = []
    for u, v, r in day2:
        reversed_day2_graph.append((v, u, 1.0 / r))
    m2_rev = compute_max_product(reversed_day2_graph, initial)
    
    max_product = 1.0
    for c in all_c:
        product = m1[c] * m2_rev[c]
        if product > max_product:
            max_product = product
    print(max_product)

if __name__ == '__main__':
    main()