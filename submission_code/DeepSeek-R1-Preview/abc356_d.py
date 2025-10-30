MOD = 998244353

N, M = map(int, input().split())

result = 0

for i in range(61):
    mask = 1 << i
    if not (M & mask):
        continue
    two_i_plus_1 = mask << 1
    full_cycles = (N + 1) // two_i_plus_1
    count = full_cycles * mask
    remaining = (N + 1) % two_i_plus_1
    if remaining > mask:
        count += remaining - mask
    result += count

print(result % MOD)