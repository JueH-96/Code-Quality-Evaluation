n = int(input())
x = list(map(int, input().split()))

ans = sum(x)
for i in range(1 << (n - 3)):
    cur_x = x[:]
    cur_ans = sum(x)
    for j in range(n - 3):
        if (i >> j) & 1:
            l = j
            r = j + 3
            m = (cur_x[l] + cur_x[r]) / 2
            cur_x[l + 1] = m + (m - cur_x[l + 1])
            cur_x[l + 2] = m - (cur_x[l + 2] - m)
            cur_ans = sum(cur_x)
    ans = min(ans, cur_ans)

print(int(ans))