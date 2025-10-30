def find_cycle(N, A):
    visited = [False] * (N + 1)
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = A[current - 1]
            if current in cycle:
                idx = cycle.index(current)
                cycle = cycle[idx:]
                return cycle
    return []

# Read input
N = int(input())
A = list(map(int, input().split()))

# Find the cycle
cycle = find_cycle(N, A)

# Print the result
print(len(cycle))
print(' '.join(map(str, cycle)))