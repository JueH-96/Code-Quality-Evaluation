# YOUR CODE HERE
n, k = map(int, input().split())
p = list(map(int, input().split()))

# For each position, find what value it will have after k operations
result = [0] * n

for i in range(n):
    # Follow the permutation starting from position i+1 (1-indexed)
    curr = i + 1
    
    # Find the cycle
    cycle = []
    visited = set()
    while curr not in visited:
        visited.add(curr)
        cycle.append(curr)
        curr = p[curr - 1]  # p is 0-indexed but contains 1-indexed values
    
    # After k steps, we're at position cycle[k % len(cycle)]
    result[i] = cycle[k % len(cycle)]

# Print result
print(*result)