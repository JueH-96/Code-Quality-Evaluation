n = int(input())
ans = 0
prev_t = 0
for _ in range(n):
    t, v = map(int, input().split())
    ans += max(v - (t - prev_t), 0)
    prev_t = t
print(max(0, ans - (100 - prev_t)))