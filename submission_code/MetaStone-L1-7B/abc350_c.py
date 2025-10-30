n = int(input())
A = list(map(int, input().split()))
A = [x - 1 for x in A]  # Convert to 0-based index

visited = [False] * n
swaps = []

for i in range(n):
    if not visited[i] and A[i] != i:
        current = i
        cycle = []
        while True:
            cycle.append(current)
            visited[current] = True
            current = A[current]
            if current == i:
                break
        # Add swaps for this cycle
        for j in range(len(cycle) - 1):
            swaps.append((cycle[j], cycle[j + 1]))

print(len(swaps))
for s in swaps:
    print(s[0] + 1, s[1] + 1)