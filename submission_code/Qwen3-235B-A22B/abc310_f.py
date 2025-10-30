MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = list(map(int, input().split()))
    n = data[0]
    A = data[1:n+1]
    
    product_A = 1
    for a in A:
        product_A = (product_A * a) % MOD
    
    inv_product_A = pow(product_A, MOD-2, MOD)
    
    # Initialize DP table
    dp_prev = [0] * (1 << 11)
    dp_prev[1] = 1  # Initially, only sum 0 is present
    
    for a_i in A:
        c1 = 1 if a_i >= 10 else 0
        c2 = max(0, a_i - 10)
        group3_size = min(a_i, 9)
        group3_values = list(range(1, group3_size + 1)) if group3_size >= 1 else []
        
        dp_next = [0] * (1 << 11)
        for mask_prev in range(1 << 11):
            cnt = dp_prev[mask_prev]
            if cnt == 0:
                continue
            
            # Contribution from values > 10 (c2)
            dp_next[mask_prev] = (dp_next[mask_prev] + cnt * c2) % MOD
            
            # Contribution from value == 10 (c1)
            mask_with_10 = mask_prev | (1 << 10)
            dp_next[mask_with_10] = (dp_next[mask_with_10] + cnt * c1) % MOD
            
            # Contribution from values < 10 (group3_values)
            for v in group3_values:
                shifted = (mask_prev << v) & ((1 << 11) - 1)
                new_mask = mask_prev | shifted
                dp_next[new_mask] = (dp_next[new_mask] + cnt) % MOD
        
        dp_prev = dp_next
    
    # Calculate invalid count (no subset sum of 10)
    invalid = 0
    for mask in range(1 << 11):
        if not (mask & (1 << 10)):
            invalid = (invalid + dp_prev[mask]) % MOD
    
    valid = (product_A - invalid) % MOD
    ans = (valid * inv_product_A) % MOD
    print(ans)

if __name__ == "__main__":
    main()