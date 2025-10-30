# YOUR CODE HERE
import itertools

def read_input():
    N = int(input())
    M_G = int(input())
    G = [[0]*N for _ in range(N)]
    for _ in range(M_G):
        u, v = map(int, input().split())
        G[u-1][v-1] = G[v-1][u-1] = 1
    M_H = int(input())
    H = [[0]*N for _ in range(N)]
    for _ in range(M_H):
        u, v = map(int, input().split())
        H[u-1][v-1] = H[v-1][u-1] = 1
    A = []
    for i in range(N-1):
        A.append(list(map(int, input().split())))
    return N, G, H, A

def is_isomorphic(G, H, perm):
    for i in range(len(G)):
        for j in range(len(G)):
            if G[i][j] != H[perm[i]][perm[j]]:
                return False
    return True

def calculate_cost(G, H, A, perm):
    cost = 0
    for i in range(len(G)):
        for j in range(i+1, len(G)):
            if G[i][j] != H[perm[i]][perm[j]]:
                cost += A[min(i, j)][max(i, j) - 1]
    return cost

def solve(N, G, H, A):
    min_cost = float('inf')
    for perm in itertools.permutations(range(N)):
        if is_isomorphic(G, H, perm):
            cost = calculate_cost(G, H, A, perm)
            min_cost = min(min_cost, cost)
    return min_cost

N, G, H, A = read_input()
print(solve(N, G, H, A))