base_str = 'wbwbwwbwbwbw'

W, B = map(int, input().split())
L = W + B

found = False
for s in range(12):
    full_cycles = L // 12
    rem = L % 12
    cnt_w = 0
    for i in range(rem):
        pos = (s + i) % 12
        if base_str[pos] == 'w':
            cnt_w += 1
    total_w = full_cycles * 7 + cnt_w
    total_b = full_cycles * 5 + (rem - cnt_w)
    if total_w == W and total_b == B:
        found = True
        break

print("Yes" if found else "No")