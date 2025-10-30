from collections import deque

def check_valid(n, testimonies, confused):
    # Build constraint graph based on confused assignment
    adj = [[] for _ in range(n)]
    
    for a, b, c in testimonies:
        a -= 1
        b -= 1
        # If confused[A] == C: A and B must have same type
        # If confused[A] != C: A and B must have different types
        if confused[a] == c:
            adj[a].append((b, 0))  # same type
            adj[b].append((a, 0))
        else:
            adj[a].append((b, 1))  # different type
            adj[b].append((a, 1))
    
    # Check if we can 2-color the graph with constraints
    color = [-1] * n
    
    for start in range(n):
        if color[start] == -1:
            queue = deque([start])
            color[start] = 0
            
            while queue:
                u = queue.popleft()
                for v, constraint_type in adj[u]:
                    if constraint_type == 0:  # same color
                        expected = color[u]
                    else:  # different color
                        expected = 1 - color[u]
                    
                    if color[v] == -1:
                        color[v] = expected
                        queue.append(v)
                    elif color[v] != expected:
                        return False
    
    return True

def solve():
    n, m = map(int, input().split())
    testimonies = []
    
    for _ in range(m):
        a, b, c = map(int, input().split())
        testimonies.append((a, b, c))
    
    # Try different strategies to find valid confused assignment
    
    # Strategy 1: No one confused
    confused = [0] * n
    if check_valid(n, testimonies, confused):
        print(''.join(map(str, confused)))
        return
    
    # Strategy 2: Everyone confused
    confused = [1] * n
    if check_valid(n, testimonies, confused):
        print(''.join(map(str, confused)))
        return
    
    # Strategy 3: Only consider villagers involved in testimonies
    involved = set()
    for a, b, c in testimonies:
        involved.add(a - 1)
        involved.add(b - 1)
    
    involved_list = list(involved)
    
    # If few villagers involved, try all combinations
    if len(involved_list) <= 20:
        for mask in range(1 << len(involved_list)):
            confused = [0] * n
            for i, villager in enumerate(involved_list):
                if (mask >> i) & 1:
                    confused[villager] = 1
            
            if check_valid(n, testimonies, confused):
                print(''.join(map(str, confused)))
                return
    else:
        # For many villagers, use heuristics
        import random
        
        # Try some patterns
        patterns = [
            lambda i: i % 2,
            lambda i: (i // 10) % 2,
            lambda i: i < n // 2,
            lambda i: i % 3 == 0,
            lambda i: i % 3 == 1,
            lambda i: i % 3 == 2,
        ]
        
        for pattern in patterns:
            confused = [pattern(i) for i in range(n)]
            if check_valid(n, testimonies, confused):
                print(''.join(map(str, confused)))
                return
        
        # Random search focusing on involved villagers
        for _ in range(50000):
            confused = [0] * n
            for v in involved_list:
                confused[v] = random.randint(0, 1)
            
            if check_valid(n, testimonies, confused):
                print(''.join(map(str, confused)))
                return
    
    print(-1)

solve()