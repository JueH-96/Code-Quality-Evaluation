# YOUR CODE HERE
from itertools import permutations

def read_input():
    import sys
    N = int(input())
    MG = int(input())
    G = [[] for _ in range(N)]
    for _ in range(MG):
        u, v = map(int, input().split())
        G[u-1].append(v-1)
        G[v-1].append(u-1)
    MH = int(input())
    H = [[] for _ in range(N)]
    for _ in range(MH):
        a, b = map(int, input().split())
        H[a-1].append(b-1)
        H[b-1].append(a-1)
    A = [list(map(int, input().split())) for _ in range(N-1)]
    return N, G, H, A

def cost_to_make_isomorphic(N, G, H, A):
    min_cost = float('inf')
    for perm in permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i+1, N):
                if (i, j) not in G and (perm[i], perm[j]) in H:
                    cost += A[i][j]
                elif (i, j) in G and (perm[i], perm[j]) not in H:
                    cost += A[i][j]
        min_cost = min(min_cost, cost)
    return min_cost

N, G, H, A = read_input()
print(cost_to_make_isomorphic(N, G, H, A))