n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed
a = [x - 1 for x in a]

total = 0
for u in range(n):
    visited = set()
    current = u
    
    # Follow the path until we revisit a vertex
    while current not in visited:
        visited.add(current)
        current = a[current]
    
    total += len(visited)

print(total)