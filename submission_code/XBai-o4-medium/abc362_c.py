n = int(input())
lr = []
sum_L = 0
sum_R = 0
for _ in range(n):
    l, r = map(int, input().split())
    lr.append((l, r))
    sum_L += l
    sum_R += r

if not (sum_L <= 0 <= sum_R):
    print("No")
else:
    x = [l for l, r in lr]
    remaining = -sum_L
    for i in range(n):
        current_l = x[i]
        current_r = lr[i][1]
        max_add = current_r - current_l
        add_this = min(max_add, remaining)
        x[i] += add_this
        remaining -= add_this
        if remaining == 0:
            break
    print("Yes")
    print(' '.join(map(str, x)))