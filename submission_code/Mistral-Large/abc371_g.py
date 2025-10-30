import sys
input = sys.stdin.read

def find_lexicographically_smallest_permutation():
    data = input().split()
    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    A = list(map(int, data[N+1:2*N+1]))

    # Convert P and A to 0-indexed
    P = [x - 1 for x in P]
    A = [x - 1 for x in A]

    # Find the cycles in P
    visited = [False] * N
    cycles = []

    for i in range(N):
        if not visited[i]:
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = P[x]
            cycles.append(cycle)

    # Rotate each cycle to find the lexicographically smallest permutation
    for cycle in cycles:
        min_rotation = min(range(len(cycle)), key=lambda i: A[cycle[i:] + cycle[:i]])
        rotated_cycle = cycle[min_rotation:] + cycle[:min_rotation]
        for j, idx in enumerate(rotated_cycle):
            A[idx] = A[cycle[(j + min_rotation) % len(cycle)]]

    # Convert A back to 1-indexed
    A = [x + 1 for x in A]

    print(' '.join(map(str, A)))

find_lexicographically_smallest_permutation()