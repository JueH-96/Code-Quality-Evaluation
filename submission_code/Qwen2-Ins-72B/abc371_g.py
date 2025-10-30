from collections import deque
N = int(input())
P = [int(x) for x in input().split()]
A = [int(x) for x in input().split()]
P_inv = [0] * N
for i, p in enumerate(P):
    P_inv[p-1] = i
A_rank = sorted(range(N), key=lambda i: A[i])
P_inv_rank = sorted(range(N), key=lambda i: P_inv[i])
group = [0] * N
group_id = 0
for i in range(N):
    if group[i] == 0:
        group_id += 1
        q = deque([i])
        while q:
            j = q.popleft()
            group[j] = group_id
            q.append(P_inv_rank[j])
A_rank_group = [[] for _ in range(group_id+1)]
for i in A_rank:
    A_rank_group[group[i]].append(A[i])
for g in A_rank_group:
    g.sort(reverse=True)
ans = []
for i in P_inv_rank:
    ans.append(A_rank_group[group[i]].pop())
print(*ans)