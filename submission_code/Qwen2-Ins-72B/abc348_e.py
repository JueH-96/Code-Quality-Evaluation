import sys
input = sys.stdin.readline

def dfs1(v, p, d):
    depth[v] = d
    for u in G[v]:
        if u == p: continue
        dfs1(u, v, d+1)

def dfs2(v, p, s):
    score[v] = s
    for u in G[v]:
        if u == p: continue
        dfs2(u, v, s + (N-depth[u])*C[v] - C[u]*depth[u])

N = int(input())
G = [[] for _ in range(N)]
for _ in range(N-1):
    a, b = map(int, input().split())
    G[a-1].append(b-1)
    G[b-1].append(a-1)
C = list(map(int, input().split()))

depth = [0]*N
dfs1(0, -1, 0)
score = [0]*N
dfs2(0, -1, sum(C))
print(min(score))