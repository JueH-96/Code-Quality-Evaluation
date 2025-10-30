from collections import deque

def min_cost(strings):
    n = len(strings)
    costs = [float('inf')] * n
    queue = deque([(s, 0) for s in strings])
    
    while queue:
        s, cost = queue.popleft()
        if cost < costs[n-len(s)]:
            costs[n-len(s)] = cost
            for i in range(len(s)):
                queue.append((s[:i] + s[i+1:], cost + 1))
                queue.append((s + 'a', cost + 1))
    
    return costs

# Read input
n = int(input())
strings = [input() for _ in range(n)]

# Solve and print output
for cost in min_cost(strings):
    print(cost)