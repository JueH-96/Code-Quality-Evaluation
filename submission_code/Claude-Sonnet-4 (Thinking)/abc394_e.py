from collections import deque

n = int(input())
graph = []
for i in range(n):
    row = input().strip()
    graph.append(row)

def is_palindrome(s):
    return s == s[::-1]

result = [[-1] * n for _ in range(n)]

max_depth = 200  # Reasonable bound to prevent excessive computation

for start in range(n):
    # BFS from start
    queue = deque([(start, 0, "")])  # (current_vertex, path_length, current_string)
    visited = set()
    visited.add((start, ""))
    
    while queue:
        curr_vertex, path_length, curr_string = queue.popleft()
        
        # Check if current string is a palindrome
        if is_palindrome(curr_string):
            if result[start][curr_vertex] == -1:
                result[start][curr_vertex] = path_length
        
        # Limit search depth
        if path_length >= max_depth:
            continue
        
        # Expand to neighbors
        for next_vertex in range(n):
            edge_label = graph[curr_vertex][next_vertex]
            if edge_label != '-':
                new_string = curr_string + edge_label
                if (next_vertex, new_string) not in visited:
                    visited.add((next_vertex, new_string))
                    queue.append((next_vertex, path_length + 1, new_string))

for i in range(n):
    print(' '.join(map(str, result[i])))