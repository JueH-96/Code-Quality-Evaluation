mod = 998244353
n, m = map(int, input().split())
result = 0
for i in range(61):  # Check each bit from 0 to 60
    if (m >> i) & 1:
        cycle = 1 << (i + 1)
        half = 1 << i
        full_cycles = (n + 1) // cycle
        remainder = (n + 1) % cycle
        cnt = full_cycles * half + max(0, remainder - half)
        result += cnt
print(result % mod)