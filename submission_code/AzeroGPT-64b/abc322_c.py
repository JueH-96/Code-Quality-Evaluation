import sys
input = sys.stdin.readline

N, M = map(int, input().split())
A = list(map(int, input().split()))

# 線分木を用意
S = 1 << (N.bit_length() + 1)
tree = [-1] * S

# 線分木にデータの更新を行う
def update(i, x):
    i += S // 2
    tree[i] = x
    i //= 2
    while i > 0:
        tree[i] = tree[2 * i] if tree[2 * i] >= 0 else tree[2 * i + 1]
        i //= 2

# 線分木から左側最近のデータの取得
def query_left(root, value, left=0, right=S//2, node=1):
    if right - left == 1:
        return left
    else:
        if tree[node] < value:
            return left
        mid = (left + right) // 2
        if tree[2 * node] >= value:
            return query_left(2 * node, value, left, mid)
        else:
            return query_left(2 * node + 1, value, mid, right)

# 答えの初期化
ans = [-1] * N

for a in A:
    # 線分木に現在のfireworksの日付を書き込む
    update(a, a)
    day = query_left(1, a)  # 最初にfireworksが行われる日付を取得
    if day == -1:
        day = a
    ans[day - 1] = 0

# 未更新の部分を更新
for i in range(N - 1, 0, -1):
    if ans[i] != -1: break
    ans[i - 1] = ans[i] + 1

for i in range(N):
    print(ans[i])