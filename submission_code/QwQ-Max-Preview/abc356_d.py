MOD = 998244353

N, M = map(int, input().split())

result = 0
for i in range(61):
    if (M >> i) & 1:
        cycle = 1 << (i + 1)
        full_cycles = (N + 1) // cycle
        count = full_cycles * (1 << i)
        remainder = (N + 1) % cycle
        count += max(0, remainder - (1 << i))
        result = (result + count) % MOD

print(result)