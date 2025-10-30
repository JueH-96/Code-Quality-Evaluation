from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    
    # Build the family tree - children[i] contains list of children of person i
    children = defaultdict(list)
    
    if N > 1:
        parents = list(map(int, input().split()))
        for i in range(2, N + 1):
            parent = parents[i - 2]  # parents[0] is parent of person 2
            children[parent].append(i)
    
    # Set to track all people covered by at least one insurance
    covered = set()
    
    # Process each insurance
    for _ in range(M):
        x, y = map(int, input().split())
        
        # Find all people covered by this insurance using BFS
        # Start with the person who bought the insurance
        queue = deque([(x, 0)])  # (person, generation)
        visited = set([x])
        
        while queue:
            person, gen = queue.popleft()
            covered.add(person)
            
            # If we haven't reached the maximum generation, add children
            if gen < y:
                for child in children[person]:
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, gen + 1))
    
    print(len(covered))

solve()