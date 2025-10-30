# YOUR CODE HERE
from collections import defaultdict, deque

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    sets = []
    element_to_sets = defaultdict(list)
    
    for i in range(N):
        A_i = int(data[idx])
        idx += 1
        S_i = list(map(int, data[idx:idx + A_i]))
        idx += A_i
        sets.append(set(S_i))
        for elem in S_i:
            element_to_sets[elem].append(i)
    
    if 1 == M:
        print(0)
        return
    
    # BFS to find the minimum number of operations
    queue = deque()
    visited = [False] * N
    for set_idx in element_to_sets[1]:
        queue.append((set_idx, 0))
        visited[set_idx] = True
    
    while queue:
        current_set_idx, operations = queue.popleft()
        current_set = sets[current_set_idx]
        
        if M in current_set:
            print(operations)
            return
        
        for elem in current_set:
            for neighbor_set_idx in element_to_sets[elem]:
                if not visited[neighbor_set_idx]:
                    visited[neighbor_set_idx] = True
                    queue.append((neighbor_set_idx, operations + 1))
    
    print(-1)

solve()