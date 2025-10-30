from collections import deque

def solve(N, M, X, Y, Z):
    # Check for self-loops (X_i == Y_i)
    for i in range(M):
        if X[i] == Y[i] and Z[i] != 0:
            return None  # No solution exists
    
    # Build a graph from the given equations
    graph = [[] for _ in range(N+1)]
    for i in range(M):
        x, y, z = X[i], Y[i], Z[i]
        if x != y:  # Skip self-loops (already checked above)
            graph[x].append((y, z))
            graph[y].append((x, z))
    
    A = [-1] * (N+1)  # -1 indicates that the value is not yet assigned
    
    for i in range(1, N+1):
        if A[i] == -1:  # If the value is not yet assigned
            A[i] = 0  # Arbitrarily choose 0 to minimize the sum
            queue = deque([(i, 0)])
            
            while queue:
                node, val = queue.popleft()
                
                for neighbor, z in graph[node]:
                    expected_val = val ^ z
                    
                    if A[neighbor] == -1:  # If the value is not yet assigned
                        A[neighbor] = expected_val
                        queue.append((neighbor, expected_val))
                    elif A[neighbor] != expected_val:  # If there's a conflict
                        return None  # No solution exists
    
    return A[1:]

# Main function to read input and print output
N, M = map(int, input().split())
X, Y, Z = [], [], []
for _ in range(M):
    x, y, z = map(int, input().split())
    X.append(x)
    Y.append(y)
    Z.append(z)

result = solve(N, M, X, Y, Z)
if result is None:
    print(-1)
else:
    print(" ".join(map(str, result)))