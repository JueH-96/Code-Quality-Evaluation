# YOUR CODE HERE
import sys
import threading

sys.setrecursionlimit(1 << 25)
def main():
    N, M = map(int, sys.stdin.readline().split())
    p_list = list(map(int, sys.stdin.readline().split()))
    tree = [[] for _ in range(N + 1)]  # Nodes are from 1 to N
    for i, p_i in enumerate(p_list, 2):
        tree[p_i].append(i)
    cover = [-1] * (N + 1)  # Initialize coverages to -1
    insurances = []
    for _ in range(M):
        x_i, y_i = map(int, sys.stdin.readline().split())
        if cover[x_i] < y_i:
            cover[x_i] = y_i
    total_covered = 0

    def dfs(u):
        nonlocal total_covered
        if cover[u] >= 0:
            total_covered +=1
        for v in tree[u]:
            if cover[u] >= 0:
                cover_v = cover[u]-1
                if cover[v] < cover_v:
                    cover[v] = cover_v
            dfs(v)
    dfs(1)
    print(total_covered)