# YOUR CODE HERE
n = int(input())
x = list(map(int, input().split()))
q = int(input())
tasks = []
for i in range(q):
    t, g = map(int, input().split())
    tasks.append((t - 1, g))

ans = 0
cur_pos = x[:]
for i in range(q):
    t, g = tasks[i]
    diff = g - cur_pos[t]
    ans += abs(diff)
    cur_pos[t] = g
    for j in range(n):
        if j != t:
            if cur_pos[j] > g and cur_pos[j] > cur_pos[t]:
                ans += cur_pos[j] - g
                cur_pos[j] = g
            elif cur_pos[j] < g and cur_pos[j] < cur_pos[t]:
                ans += g - cur_pos[j]
                cur_pos[j] = g

print(ans)