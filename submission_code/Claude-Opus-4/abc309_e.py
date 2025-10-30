# YOUR CODE HERE
from collections import defaultdict, deque

def solve():
    # Read input
    N, M = map(int, input().split())
    
    # Build adjacency list for the tree
    children = defaultdict(list)
    
    if N > 1:
        parents = list(map(int, input().split()))
        for i in range(len(parents)):
            parent = parents[i]
            child = i + 2  # Since parents array starts from person 2
            children[parent].append(child)
    
    # Set to store all covered people
    covered = set()
    
    # Process each insurance
    for _ in range(M):
        x, y = map(int, input().split())
        
        # BFS to find all descendants within y generations
        queue = deque([(x, 0)])  # (person, generation)
        
        while queue:
            person, gen = queue.popleft()
            covered.add(person)
            
            # If we haven't reached the generation limit, add children
            if gen < y:
                for child in children[person]:
                    queue.append((child, gen + 1))
    
    # Output the number of covered people
    print(len(covered))

solve()