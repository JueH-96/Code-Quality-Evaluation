import itertools
import sys

def read_input():
    N = int(input())
    MG = int(input())
    G = set(tuple(map(int, input().split())) for _ in range(MG))
    MH = int(input())
    H = set(tuple(map(int, input().split())) for _ in range(MH))
    A = []
    for i in range(N-1):
        A.append(list(map(int, input().split())))
    return N, G, H, A

def cost_to_make_isomorphic(N, G, H, A):
    min_cost = float('inf')
    
    for perm in itertools.permutations(range(N)):
        cost = 0
        for i in range(N):
            for j in range(i + 1, N):
                u, v = i + 1, j + 1
                pu, pv = perm[i] + 1, perm[j] + 1
                if (u, v) in G and (pu, pv) not in H:
                    cost += A[i][j - i - 1]
                elif (u, v) not in G and (pu, pv) in H:
                    cost += A[i][j - i - 1]
        min_cost = min(min_cost, cost)
    
    return min_cost

def main():
    N, G, H, A = read_input()
    print(cost_to_make_isomorphic(N, G, H, A))

if __name__ == "__main__":
    main()