n = int(input())
s = input()
t = input()

# Build the mapping
mapping = {}
for i in range(n):
    if s[i] in mapping:
        if mapping[s[i]] != t[i]:
            print(-1)
            exit()
    else:
        mapping[s[i]] = t[i]

# Build the graph (only non-trivial mappings)
graph = {c: mapping[c] for c in mapping if c != mapping[c]}

# Find all cycles using DFS
color = {}  # 0: white (unvisited), 1: gray (visiting), 2: black (visited)
num_cycles = 0

def dfs(node):
    global num_cycles
    if node not in graph:
        return
    if node in color:
        if color[node] == 1:
            num_cycles += 1
        return
    color[node] = 1
    dfs(graph[node])
    color[node] = 2

for node in graph:
    if node not in color:
        dfs(node)

# Count operations
num_edges = len(graph)
operations = num_edges + num_cycles
print(operations)