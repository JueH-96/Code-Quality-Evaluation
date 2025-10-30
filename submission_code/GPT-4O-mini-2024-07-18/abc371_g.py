def lexicographically_smallest_permutation(N, P, A):
    # Create a mapping from index to its position in P
    position = [0] * N
    for i in range(N):
        position[P[i] - 1] = i

    # Create a new array to hold the result
    result = A[:]
    visited = [False] * N

    for i in range(N):
        if not visited[i]:
            # Start a new cycle
            cycle = []
            x = i
            while not visited[x]:
                visited[x] = True
                cycle.append(A[x])
                x = position[x]

            # Sort the cycle to get the smallest lexicographical order
            cycle.sort()
            # Place the sorted values back into the result
            for j, val in enumerate(cycle):
                result[i] = cycle[j]
                i = position[i]

    return result

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
P = list(map(int, data[1].split()))
A = list(map(int, data[2].split()))

result = lexicographically_smallest_permutation(N, P, A)
print(' '.join(map(str, result)))