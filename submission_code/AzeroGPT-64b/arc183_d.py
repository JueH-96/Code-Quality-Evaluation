import sys
input = sys.stdin.buffer.readline

N = int(input())
link = [[] for _ in range(N)]
for _ in range(N-1):
    a, b = map(int, input().split())
    a -= 1
    b -= 1
    link[a].append(b)
    link[b].append(a)

# 各グループごとの葉の候補を列挙
groups = [[] for _ in range(N//2)]
for i in range(N):
    if len(link[i]) == 1:
        if i%2 == 0:
            groups[i//2].append(i)
        else:
            groups[i//2-1].append(i)

# 各グループごとに Fredericksen's algorithm を使う
up = [-1] * N
children = [[] for _ in range(N)]
for s in range(0, N, 2):
    u = s
    d = 0
    while d < len(link[u]):
        v = link[u][d]
        up[v] = u
        children[u].append(v)
        d += 1
        u = v

# 深さ優先探索, 適当に２つ選ぶ
answer = []
mapping = [-1] * N
for i, (a, b) in enumerate([0, 1], start=1):
    stack = [a, b]
    mapping[b] = len(answer)
    answer.append(b)
    while stack:
        v = stack.pop()
        for u in children[v]:
            stack.append(u)
            mapping[u] = len(answer)
            answer.append(u)
for i in range(len(answer)//2):
    print(answer[2*i]+1, answer[2*i+1]+1)