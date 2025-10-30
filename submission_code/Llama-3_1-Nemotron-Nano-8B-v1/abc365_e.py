n, *rest = map(int, open(0).read().split())
A = rest[:n]

prefix_xor = [0]
current_xor = 0
for a in A:
    current_xor ^= a
    prefix_xor.append(current_xor)

max_bit = 30
total_bit = [0] * (max_bit + 1)
cnt0 = [1] * (max_bit + 1)
cnt1 = [0] * (max_bit + 1)

for j in range(2, len(prefix_xor)):
    x = prefix_xor[j]
    for k in range(max_bit + 1):
        b = (x >> k) & 1
        if b == 0:
            total_bit[k] += cnt1[k]
        else:
            total_bit[k] += cnt0[k]
    x_prev = prefix_xor[j-1]
    for k in range(max_bit + 1):
        b = (x_prev >> k) & 1
        if b == 0:
            cnt0[k] += 1
        else:
            cnt1[k] += 1

ans = 0
for k in range(max_bit + 1):
    ans += total_bit[k] * (1 << k)
print(ans)