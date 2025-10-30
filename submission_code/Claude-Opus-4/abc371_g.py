# YOUR CODE HERE
def find_cycles(P):
    n = len(P)
    visited = [False] * n
    cycles = []
    
    for i in range(n):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j] - 1  # Convert to 0-indexed
            cycles.append(cycle)
    
    return cycles

def find_lexicographically_smallest(n, P, A):
    # Convert to 0-indexed
    P = [p - 1 for p in P]
    A = [a - 1 for a in A]
    
    # Find all cycles in P
    cycles = find_cycles(P)
    
    # Result array
    result = [0] * n
    
    # For each cycle, find the best rotation
    for cycle in cycles:
        # Get values in this cycle
        values = [A[i] for i in cycle]
        
        # Try all rotations and find the lexicographically smallest
        best_rotation = values[:]
        for rotation in range(len(cycle)):
            rotated = values[rotation:] + values[:rotation]
            
            # Check if this rotation is lexicographically smaller
            is_smaller = False
            for i in range(len(cycle)):
                if rotated[i] < best_rotation[i]:
                    is_smaller = True
                    break
                elif rotated[i] > best_rotation[i]:
                    break
            
            if is_smaller:
                best_rotation = rotated[:]
        
        # Place the best rotation back
        for i, pos in enumerate(cycle):
            result[pos] = best_rotation[i]
    
    # Convert back to 1-indexed and return
    return [r + 1 for r in result]

# Read input
n = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Solve and print result
result = find_lexicographically_smallest(n, P, A)
print(' '.join(map(str, result)))