n, m = map(int, input().split())
strings = [input().strip() for _ in range(n)]

def hamming(s, t):
    if len(s) != len(t):
        return -1  # This should not happen as per problem constraints
    count = 0
    for c1, c2 in zip(s, t):
        if c1 != c2:
            count += 1
    return count

# Build adjacency list
adj = [[] for _ in range(n)]
for i in range(n):
    for j in range(n):
        if i != j and hamming(strings[i], strings[j]) == 1:
            adj[i].append(j)

def dfs(current, visited):
    if visited == (1 << n) - 1:
        return True
    for neighbor in adj[current]:
        if not (visited & (1 << neighbor)):
            if dfs(neighbor, visited | (1 << neighbor)):
                return True
    return False

found = False
for i in range(n):
    if dfs(i, 1 << i):
        found = True
        break

print("Yes" if found else "No")