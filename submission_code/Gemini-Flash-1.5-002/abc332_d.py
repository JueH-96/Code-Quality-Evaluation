# YOUR CODE HERE
from itertools import permutations

def solve():
    H, W = map(int, input().split())
    A = []
    for _ in range(H):
        A.append(list(map(int, input().split())))
    B = []
    for _ in range(H):
        B.append(list(map(int, input().split())))

    q = [(A, 0)]
    visited = {tuple(map(tuple, A))}
    
    while q:
        curr_A, dist = q.pop(0)
        if curr_A == B:
            print(dist)
            return

        # Row swaps
        for i in range(H - 1):
            next_A = [row[:] for row in curr_A]
            next_A[i], next_A[i+1] = next_A[i+1], next_A[i]
            next_A_tuple = tuple(map(tuple, next_A))
            if next_A_tuple not in visited:
                visited.add(next_A_tuple)
                q.append((next_A, dist + 1))

        # Column swaps
        for i in range(W - 1):
            next_A = [row[:] for row in curr_A]
            for j in range(H):
                next_A[j][i], next_A[j][i+1] = next_A[j][i+1], next_A[j][i]
            next_A_tuple = tuple(map(tuple, next_A))
            if next_A_tuple not in visited:
                visited.add(next_A_tuple)
                q.append((next_A, dist + 1))

    print(-1)

solve()