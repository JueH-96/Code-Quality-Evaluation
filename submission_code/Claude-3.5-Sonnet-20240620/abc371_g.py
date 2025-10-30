# YOUR CODE HERE
def find_smallest_permutation(N, P, A):
    cycles = []
    visited = [False] * (N + 1)
    
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current - 1]
            cycles.append(cycle)
    
    result = [0] * N
    for cycle in cycles:
        min_value = min(A[i - 1] for i in cycle)
        min_index = A.index(min_value)
        for i, pos in enumerate(cycle):
            result[pos - 1] = A[(min_index - cycle.index(pos)) % len(cycle)]
    
    return result

N = int(input())
P = list(map(int, input().split()))
A = list(map(int, input().split()))

smallest_permutation = find_smallest_permutation(N, P, A)
print(*smallest_permutation)