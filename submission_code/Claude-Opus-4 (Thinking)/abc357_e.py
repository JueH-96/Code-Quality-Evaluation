N = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed
for i in range(N):
    a[i] -= 1

total_count = 0
for u in range(N):
    visited = set()
    current = u
    while current not in visited:
        visited.add(current)
        current = a[current]
    total_count += len(visited)

print(total_count)