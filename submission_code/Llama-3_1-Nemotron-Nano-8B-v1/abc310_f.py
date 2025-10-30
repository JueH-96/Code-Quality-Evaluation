MOD = 998244353

def main():
    import sys
    from collections import defaultdict

    N, *rest = list(map(int, sys.stdin.read().split()))
    A = rest[:N]
    
    product = 1
    for a in A:
        product = product * a % MOD
    
    current_dp = defaultdict(int)
    current_dp[(False, 1)] = 1  # (has_found, mask), mask is 1 << 0 (sum 0)
    
    for a in A:
        new_dp = defaultdict(int)
        for (hf, mask), cnt in current_dp.items():
            if hf:
                new_dp[(True, mask)] = (new_dp[(True, mask)] + cnt * a) % MOD
            else:
                # Case 1: not include the current die
                new_dp[(hf, mask)] = (new_dp[(hf, mask)] + cnt) % MOD
                
                # Case 2: include the current die
                for s in range(11):
                    if (mask & (1 << s)) == 0:
                        continue
                    possible_x = min(a, 10 - s)
                    if possible_x < 1:
                        continue
                    count_x_10 = 1 if (10 - s) <= possible_x else 0
                    count_x_other = possible_x - count_x_10
                    
                    # Add to has_found
                    new_dp[(True, 0)] = (new_dp[(True, 0)] + cnt * count_x_10) % MOD
                    
                    # Add to new_mask for others
                    start = s + 1
                    end = s + possible_x
                    if count_x_10:
                        if end >= 10:
                            end = 9
                    if start > end:
                        continue
                    new_mask_part = ((1 << (end + 1)) - 1) - ((1 << start) - 1)
                    new_mask = mask | new_mask_part
                    new_dp[(False, new_mask)] = (new_dp[(False, new_mask)] + cnt * count_x_other) % MOD
        current_dp = new_dp
    
    favorable = 0
    for (hf, _), cnt in current_dp.items():
        if hf:
            favorable = (favorable + cnt) % MOD
    
    inv_product = pow(product, MOD - 2, MOD)
    ans = (favorable * inv_product) % MOD
    print(ans)

if __name__ == "__main__":
    main()