import bisect

W, H = map(int, input().split())
N = int(input())
PQ = [tuple(map(int, input().split())) for _ in range(N)]
A = int(input())
a = list(map(int, input().split()))
B = int(input())
b = list(map(int, input().split()))

x = [0] + a + [W]
y = [0] + b + [H]

x_idx = [bisect.bisect(x, p) for p, q in PQ]
y_idx = [bisect.bisect(y, q) for p, q in PQ]

x_cnt = [0] * (A + 2)
y_cnt = [0] * (B + 2)

for i in x_idx:
    x_cnt[i] += 1

for i in y_idx:
    y_cnt[i] += 1

x_sums = list(accumulate(x_cnt))
y_sums = list(accumulate(y_cnt))

min_cnt = max_cnt = 0

for i in range(A + 1):
    for j in range(B + 1):
        cnt = x_sums[i + 1] - x_sums[i] + y_sums[j + 1] - y_sums[j] - (x_sums[i + 1] - x_sums[i]) * (y_sums[j + 1] - y_sums[j])
        min_cnt = max(min_cnt, cnt)
        max_cnt += cnt

print(min_cnt, max_cnt)