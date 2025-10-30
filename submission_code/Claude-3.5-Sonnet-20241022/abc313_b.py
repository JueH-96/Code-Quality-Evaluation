def find_strongest(N, M, edges):
    # Create adjacency list to represent who is stronger than whom
    graph = [[] for _ in range(N+1)]
    for a, b in edges:
        graph[a].append(b)
    
    # For each person, check if they could be strongest
    possible_strongest = []
    
    for person in range(1, N+1):
        # Do BFS from this person to find everyone they're stronger than
        visited = [False] * (N+1)
        visited[person] = True
        queue = [person]
        reachable = set()
        
        while queue:
            curr = queue.pop(0)
            for next_person in graph[curr]:
                if not visited[next_person]:
                    visited[next_person]= True
                    queue.append(next_person)
                    reachable.add(next_person)
        
        # If this person can reach N-1 others (everyone else), they could be strongest
        if len(reachable) == N-1:
            possible_strongest.append(person)
    
    # If exactly one person could be strongest, return them
    if len(possible_strongest) == 1:
        return possible_strongest[0]
    return -1

# Read input
N, M = map(int, input().split())
edges = []
for _ in range(M):
    a, b = map(int, input().split())
    edges.append((a, b))

# Print result
print(find_strongest(N, M, edges))