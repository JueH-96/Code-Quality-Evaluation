# YOUR CODE HERE
def find_smallest_permutation(N, P, A):
    # Create a mapping from each element to its position in P
    P_inv = [0] * (N + 1)
    for i in range(N):
        P_inv[P[i]] = i
    
    # Create a list of cycles
    visited = [False] * N
    cycles = []
    
    for i in range(N):
        if not visited[i]:
            cycle = []
            while not visited[i]:
                visited[i] = True
                cycle.append(i)
                i = P_inv[A[i]]
            cycles.append(cycle)
    
    # For each cycle, find the lexicographically smallest permutation
    for cycle in cycles:
        min_perm = A[:]
        for _ in range(len(cycle)):
            new_perm = A[:]
            for j in cycle:
                new_perm[j] = A[P_inv[j]]
            if new_perm < min_perm:
                min_perm = new_perm
            A = new_perm[:]
    
    return A

# Read input
N = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

# Solve the problem
result = find_smallest_permutation(N, P, A)

# Print the result
print(" ".join(map(str, result)))