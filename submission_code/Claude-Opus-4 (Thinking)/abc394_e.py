from collections import deque

n = int(input())
graph = []
for _ in range(n):
    graph.append(input().strip())

def find_shortest_palindrome(start, end):
    queue = deque([(start, end, 0)])
    visited = set([(start, end)])
    
    while queue:
        u, v, length = queue.popleft()
        
        # Check if we've found a palindrome (even length)
        if u == v:
            return length
        
        # Check for odd-length palindrome
        if graph[u][v] != '-':
            return length + 1
        
        # Expand: take one step from both ends with the same label
        for i in range(n):
            if graph[u][i] != '-':
                label = graph[u][i]
                # Find edges into v with the same label
                for k in range(n):
                    if graph[k][v] != '-' and graph[k][v] == label:
                        new_state = (i, k)
                        if new_state not in visited:
                            visited.add(new_state)
                            queue.append((i, k, length + 2))
    
    return -1

# Calculate and print results
for i in range(n):
    row = []
    for j in range(n):
        row.append(find_shortest_palindrome(i, j))
    print(' '.join(map(str, row)))