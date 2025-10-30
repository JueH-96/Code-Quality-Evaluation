from collections import deque

def solve():
    # Read input
    N, M = map(int, input().split())
    parent_values = list(map(int, input().split()))
    
    # Create a parent list, where parent[i] is the parent of person i
    parent = [0] * (N+1)  # 0 is a placeholder for person 1, who doesn't have a parent
    for i in range(2, N+1):
        parent[i] = parent_values[i-2]
    
    # Build the family tree (parent -> children)
    children = [[] for _ in range(N+1)]
    for i in range(2, N+1):
        children[parent[i]].append(i)
    
    # Determine the people covered by each insurance
    covered = set()
    for _ in range(M):
        x, y = map(int, input().split())
        
        queue = deque([(x, 0)])  # (person, generation)
        visited = set([x])
        
        while queue:
            person, generation = queue.popleft()
            
            covered.add(person)
            
            if generation < y:
                for child in children[person]:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, generation + 1))
    
    return len(covered)

print(solve())