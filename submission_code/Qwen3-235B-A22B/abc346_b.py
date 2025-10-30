W, B = map(int, input().split())
s_base = 'wbwbwwbwbwbw'
pre_w = [0] * 13
pre_b = [0] * 13

for i in range(12):
    pre_w[i+1] = pre_w[i] + (1 if s_base[i] == 'w' else 0)
    pre_b[i+1] = pre_b[i] + (1 if s_base[i] == 'b' else 0)

L = W + B
found = False

for s in range(12):
    l_start = 12 - s
    if L <= l_start:
        sum_w = pre_w[s + L] - pre_w[s]
        sum_b = pre_b[s + L] - pre_b[s]
    else:
        remaining = L - l_start
        full_cycles = remaining // 12
        end_remain = remaining % 12
        sum_w = pre_w[12] - pre_w[s] + full_cycles * 7
        sum_b = pre_b[12] - pre_b[s] + full_cycles * 5
        sum_w += pre_w[end_remain] - pre_w[0]
        sum_b += pre_b[end_remain] - pre_b[0]
    if sum_w == W and sum_b == B:
        found = True
        break

print("Yes" if found else "No")