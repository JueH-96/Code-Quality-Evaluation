n, k = map(int, input().split())
p = list(map(int, input().split()))
p = [0] + p  # Convert to 1-based indexing

visited = [False] * (n + 1)
result = [0] * (n + 1)

for i in range(1, n + 1):
    if not visited[i]:
        cycle = []
        j = i
        while not visited[j]:
            visited[j] = True
            cycle.append(j)
            j = p[j]  # Follow the permutation to build the cycle
        
        L = len(cycle)
        m = pow(2, k, L)  # Efficiently compute 2^k mod L
        
        for idx in range(L):
            original = cycle[idx]
            new_idx = (idx + m) % L
            result[original] = cycle[new_idx]

print(' '.join(map(str, result[1:n+1])))