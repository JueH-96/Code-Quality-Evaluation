N, M = map(int, input().split())
mod = 998244353
result = 0
for i in range(60):
    if (M >> i) & 1:
        mask = 1 << i
        cycle = mask << 1
        full_cycles = (N + 1) // cycle
        rem = (N + 1) % cycle
        cnt = full_cycles * mask
        if rem > 0:
            cnt += max(0, rem - mask)
        result += cnt
print(result % mod)