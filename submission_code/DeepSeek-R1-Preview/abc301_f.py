from collections import defaultdict

MOD = 998244353

S = input().strip()

uppercase = [chr(ord('A') + i) for i in range(26)]

current_dp = defaultdict(int)
current_dp[(0, None)] = 1

for char in S:
    new_dp = defaultdict(int)
    for (stage, c_prev), count in current_dp.items():
        if stage == 4:
            continue
        if char == '?':
            # Process uppercase case
            for c in uppercase:
                if stage == 0:
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 1:
                    if c == c_prev:
                        new_dp[(2, c_prev)] = (new_dp[(2, c_prev)] + count) % MOD
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 2:
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 3:
                    new_dp[(4, None)] = (new_dp[(4, None)] + count) % MOD
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
            # Process lowercase case
            if stage == 0:
                new_dp[(0, None)] = (new_dp[(0, None)] + count * 26) % MOD
            elif stage == 1:
                new_dp[(3, None)] = (new_dp[(3, None)] + count * 26) % MOD
            elif stage == 2:
                new_dp[(3, None)] = (new_dp[(3, None)] + count * 26) % MOD
            elif stage == 3:
                new_dp[(3, None)] = (new_dp[(3, None)] + count * 26) % MOD
        else:
            is_upper = char.isupper()
            if is_upper:
                c = char
                if stage == 0:
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 1:
                    if c == c_prev:
                        new_dp[(2, c_prev)] = (new_dp[(2, c_prev)] + count) % MOD
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 2:
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
                elif stage == 3:
                    new_dp[(4, None)] = (new_dp[(4, None)] + count) % MOD
                    new_dp[(1, c)] = (new_dp[(1, c)] + count) % MOD
            else:
                if stage == 0:
                    new_dp[(0, None)] = (new_dp[(0, None)] + count) % MOD
                elif stage == 1:
                    new_dp[(3, None)] = (new_dp[(3, None)] + count) % MOD
                elif stage == 2:
                    new_dp[(3, None)] = (new_dp[(3, None)] + count) % MOD
                elif stage == 3:
                    new_dp[(3, None)] = (new_dp[(3, None)] + count) % MOD
    current_dp = new_dp

total = 0
for (stage, c_prev), count in current_dp.items():
    if stage != 4:
        total = (total + count) % MOD
print(total)