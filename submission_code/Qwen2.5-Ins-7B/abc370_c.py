# YOUR CODE HERE
from collections import deque

def solve(S, T):
    queue = deque([(S, 0)])
    visited = set([S])
    path = {}
    
    while queue:
        current, steps = queue.popleft()
        if current == T:
            result = []
            while current != S:
                result.append(current)
                current = path[current]
            result.append(S)
            result.reverse()
            return steps + 1, result
        
        for i in range(len(current)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_str = current[:i] + c + current[i+1:]
                if next_str not in visited and next_str < current:
                    visited.add(next_str)
                    path[next_str] = current
                    queue.append((next_str, steps + 1))
                    
    return 0, []

S = input().strip()
T = input().strip()
M, X = solve(S, T)
print(M)
for x in X:
    print(x)