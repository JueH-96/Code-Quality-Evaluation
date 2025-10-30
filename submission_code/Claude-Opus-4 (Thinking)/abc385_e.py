# YOUR CODE HERE
n = int(input())
adj = [[] for _ in range(n+1)]

for _ in range(n-1):
    u, v = map(int, input().split())
    adj[u].append(v)
    adj[v].append(u)

max_kept = 0

for center in range(1, n+1):
    neighbor_leaves = []
    
    for neighbor in adj[center]:
        leaves = 0
        for v in adj[neighbor]:
            if v != center and len(adj[v]) == 1:
                leaves += 1
        if leaves > 0:
            neighbor_leaves.append(leaves)
    
    if neighbor_leaves:
        neighbor_leaves.sort(reverse=True)
        
        for x in range(1, len(neighbor_leaves) + 1):
            y = neighbor_leaves[x-1]
            kept = 1 + x + x * y
            max_kept = max(max_kept, kept)

print(n - max_kept)