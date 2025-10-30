import sys
from collections import defaultdict

MOD = 998244353

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    parent = [0] * (N + 2)
    children = defaultdict(list)
    in_degree = defaultdict(int)
    
    for i in range(1, N + 1):
        a = A[i - 1]
        if a < i:
            j = a
            if j + 1 < i:
                parent[i] = j
                children[j].append(i)
                in_degree[i] += 1
    
    tree = defaultdict(list)
    for i in range(1, N + 1):
        if parent[i] != 0:
            tree[parent[i]].append(i)
    
    size = [1] * (N + 2)
    post_order = []
    
    def dfs(u):
        for v in tree[u]:
            dfs(v)
        post_order.append(u)
    
    for i in range(1, N + 1):
        if parent[i] == 0:
            dfs(i)
    
    post_order = post_order[::-1]
    
    result = 1
    for u in post_order:
        s = len(children[u])
        if s == 0:
            continue
        result = (result * size[u]) % MOD
        if s > 1:
            result = (result * (s)) % MOD
    
    print(result % MOD)

if __name__ == '__main__':
    main()