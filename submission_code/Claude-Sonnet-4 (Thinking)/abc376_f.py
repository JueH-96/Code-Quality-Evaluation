from collections import deque

n, q = map(int, input().split())

left, right = 1, 2
total_cost = 0

for _ in range(q):
    h, t = input().split()
    t = int(t)
    
    # BFS to find minimum operations
    queue = deque([(left, right, 0)])
    visited = set([(left, right)])
    
    while queue:
        l, r, cost = queue.popleft()
        
        # Check if we reached the target
        if (h == 'L' and l == t) or (h == 'R' and r == t):
            total_cost += cost
            left, right = l, r
            break
        
        # Move left hand clockwise and counterclockwise
        for next_l in [(l % n) + 1, ((l - 2) % n) + 1]:
            if next_l != r and (next_l, r) not in visited:
                visited.add((next_l, r))
                queue.append((next_l, r, cost + 1))
        
        # Move right hand clockwise and counterclockwise  
        for next_r in [(r % n) + 1, ((r - 2) % n) + 1]:
            if next_r != l and (l, next_r) not in visited:
                visited.add((l, next_r))
                queue.append((l, next_r, cost + 1))

print(total_cost)