import sys
data = sys.stdin.read().split()
N = int(data[0])
A = list(map(int, data[1:N+1]))
indices = [[] for _ in range(N + 1)]
for i in range(N):
    val = A[i]
    pos = i + 1
    indices[val].append(pos)
total_sub = (N * (N + 1)) // 2
answer = 0
for val in range(1, N + 1):
    pos_list = indices[val]
    if not pos_list:
        continue
    p = pos_list
    sum_no_V = 0
    G_left = p[0] - 1
    sum_no_V += (G_left * (G_left + 1)) // 2
    G_right = N - p[-1]
    sum_no_V += (G_right * (G_right + 1)) // 2
    for i in range(len(p) - 1):
        G_between = p[i + 1] - p[i] - 1
        sum_no_V += (G_between * (G_between + 1)) // 2
    num_with_V = total_sub - sum_no_V
    answer += num_with_V
print(answer)