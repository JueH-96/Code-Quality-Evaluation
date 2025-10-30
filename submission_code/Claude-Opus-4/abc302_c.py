# YOUR CODE HERE
def differ_by_one(s1, s2):
    """Check if two strings differ by exactly one character"""
    if len(s1) != len(s2):
        return False
    diff_count = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            diff_count += 1
            if diff_count > 1:
                return False
    return diff_count == 1

def find_hamiltonian_path(graph, n, start, visited, path):
    """Find if there's a Hamiltonian path starting from 'start'"""
    if len(path) == n:
        return True
    
    for neighbor in graph[start]:
        if not visited[neighbor]:
            visited[neighbor] = True
            path.append(neighbor)
            
            if find_hamiltonian_path(graph, n, neighbor, visited, path):
                return True
            
            # Backtrack
            visited[neighbor] = False
            path.pop()
    
    return False

# Read input
n, m = map(int, input().split())
strings = []
for _ in range(n):
    strings.append(input().strip())

# Build adjacency list graph
graph = [[] for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j and differ_by_one(strings[i], strings[j]):
            graph[i].append(j)

# Try starting from each string
found = False
for start in range(n):
    visited = [False] * n
    visited[start] = True
    path = [start]
    
    if find_hamiltonian_path(graph, n, start, visited, path):
        found = True
        break

if found:
    print("Yes")
else:
    print("No")