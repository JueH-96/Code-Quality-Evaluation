from collections import defaultdict, deque

def solve():
    N, M = map(int, input().split())
    sets = []
    for _ in range(N):
        A = int(input())
        S = set(map(int, input().split()))
        sets.append(S)
    
    # If any set already contains 1 and M, return 0
    for s in sets:
        if 1 in s and M in s:
            return 0
    
    # Build graph of sets that share elements
    graph = defaultdict(list)
    for i in range(N):
        for j in range(i+1, N):
            if sets[i] & sets[j]:  # If sets have common elements
                graph[i].append(j)
                graph[j].append(i)
    
    # BFS to find shortest path to set containing 1 and M
    q = deque()
    visited = set()
    
    # Add all sets containing 1 or M as starting points
    for i in range(N):
        if 1 in sets[i]:
            state = (frozenset(sets[i]), {i}, 0)  # (current_set, used_indices, operations)
            q.append(state)
            visited.add((frozenset(sets[i]), frozenset({i})))
    
    while q:
        curr_set, used_indices, ops = q.popleft()
        
        # Try merging with each unused set that shares elements
        for i in used_indices:
            for j in graph[i]:
                if j not in used_indices:
                    new_set = set(curr_set) | sets[j]
                    new_used = used_indices | {j}
                    
                    state = (frozenset(new_set), frozenset(new_used))
                    if state not in visited:
                        visited.add(state)
                        if 1 in new_set and M in new_set:
                            return ops + 1
                        q.append((new_set, new_used, ops + 1))
    
    return -1

print(solve())