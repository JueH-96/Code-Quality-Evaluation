import sys

input = sys.stdin.readline

def find(target, parent):
    if parent[target] == target:
        return target
    parent[target] = find(parent[target], parent)
    return parent[target]

def union(target1, target2, parent):
    a = find(target1, parent)
    b = find(target2, parent)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

H, W = map(int, input().split())
A = [list(map(int, input().split())) for _ in range(H)]
B = [list(map(int, input().split())) for _范围  in range(H)]

matrix_A = []
matrix_B = []
for i in range(H):
    for j in range(W):
        matrix_A.append(10*j+i)
        matrix_B.append(10*j+i)

grid_parent = [i for i in range(H*W)]

for j in range(W):
    target_row = []
    for i in range(H):
        target_row.append(10*j+i)
    if target_row not in A:
        for i in range(H-1):
            if A[i][j] > A[i+1][j]:
                union(10*j+i, 10*j+i+1, grid_parent)

    target_row.reverse()
    if target_row not in A:
        for i in range(H-1):
            if A[i][j] > A[i+1][j]:
                union(10*j+i, 10*j+i+1, grid_parent)

if A != B:
    for j in range(W):
        target_row = []
        for i in range(H):
            target_row.append(10*j+i)
        if target_row not in B:
            for i in range(H-1):
                if B[i][j] > B[i+1][j]:
                    union(10*j+i, 10*j+i+1, grid_parent)
        target_row.reverse()
        if target_row not in B:
            for i in range(H-1):
                if B[i][j] > B[i+1][j]:
                    union(10*j+i, 10*j+i+1, grid_parent)

if any(find(m, grid_parent) != find(matrix_B[i], grid_parent) for i, m in enumerate(matrix_A)):
    print(-1)
else:
    ans = 0
    for i in range(H):
        for j in range(W-1):
            if A[i][j] > A[i][j+1]:
                ans += 1
            if B[i][j] > B[i][j+1]:
                ans += 1
    print(ans)