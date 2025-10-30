class Solution:
  def maximumXorProduct(self, a: int, b: int, n: int) -> int:
    MOD = 10**9 + 7

    cur_a = 0
    cur_b = 0

    # Iterate from bit 49 (MSB for 2^50-1) down to 0.
    # A higher start like 55 or 59 can be used too if numbers or n could be larger,
    # but 49 is sufficient for the given constraints.
    for k_idx in range(49, -1, -1):
        ak = (a >> k_idx) & 1  # k_idx-th bit of a
        bk = (b >> k_idx) & 1  # k_idx-th bit of b

        if k_idx >= n:
            # If k_idx >= n, x_k must be 0 (since x < 2^n).
            # So, (a XOR x)_k = a_k and (b XOR x)_k = b_k.
            if ak == 1:
                cur_a |= (1 << k_idx)
            if bk == 1:
                cur_b |= (1 << k_idx)
        else:
            # If k_idx < n, we can choose x_k = 0 or x_k = 1.
            # Let A_k = (a XOR x)_k and B_k = (b XOR x)_k.
            
            if ak == bk:
                # To maximize, set both A_k and B_k to 1.
                # If ak=bk=0, choose x_k=1 => A_k=1, B_k=1.
                # If ak=bk=1, choose x_k=0 => A_k=1, B_k=1.
                cur_a |= (1 << k_idx)
                cur_b |= (1 << k_idx)
            else: # ak != bk
                # One of ak, bk is 0 and the other is 1.
                # A_k and B_k will be one 0 and one 1, regardless of x_k choice.
                # cur_a and cur_b are values from bits > k_idx.
                
                if cur_a == cur_b:
                    # This is the first differing bit where x_k matters.
                    # Critical decision point:
                    # Option 1: A_k=1, B_k=0. A becomes dominant.
                    #   Future A gets all 1s below k_idx. Future B gets common 1s.
                    # Option 2: A_k=0, B_k=1. B becomes dominant.
                    #   Future B gets all 1s below k_idx. Future A gets common 1s.
                    
                    mask_common = 0
                    for j_low in range(k_idx - 1, -1, -1):
                        aj_low = (a >> j_low) & 1
                        bj_low = (b >> j_low) & 1
                        if aj_low == bj_low:
                            mask_common |= (1 << j_low)
                    
                    mask_all_ones = (1 << k_idx) - 1 if k_idx > 0 else 0
                    
                    # Potential values if Option 1 is chosen for bit k_idx and optimally for lower bits
                    # A gets bit k_idx, then all lower bits. B gets only common lower bits.
                    val_a_opt1 = (cur_a | (1 << k_idx)) | mask_all_ones
                    val_b_opt1 = cur_b | mask_common
                    prod1 = val_a_opt1 * val_b_opt1

                    # Potential values if Option 2 is chosen
                    # B gets bit k_idx, then all lower bits. A gets only common lower bits.
                    val_a_opt2 = cur_a | mask_common
                    val_b_opt2 = (cur_b | (1 << k_idx)) | mask_all_ones
                    prod2 = val_a_opt2 * val_b_opt2
                    
                    if prod1 >= prod2:
                        cur_a |= (1 << k_idx) # A_k=1
                        # B_k=0, so cur_b is unchanged by bit k_idx
                    else:
                        cur_b |= (1 << k_idx) # B_k=1
                        # A_k=0, so cur_a is unchanged by bit k_idx
                
                elif cur_a < cur_b:
                    # cur_a is smaller. Try to make A_k=1 to help it catch up.
                    # To make A_k=1, x_k = ak ^ 1.
                    # Then A_k = ak ^ (ak ^ 1) = 1.
                    cur_a |= (1 << k_idx)
                    # B_k = bk ^ x_k = bk ^ (ak ^ 1).
                    # Since ak != bk, ak ^ 1 = bk. So B_k = bk ^ bk = 0.
                    # cur_b is unchanged by bit k_idx.
                
                else: # cur_b < cur_a
                    # cur_b is smaller. Try to make B_k=1.
                    # To make B_k=1, x_k = bk ^ 1.
                    # Then B_k = bk ^ (bk ^ 1) = 1.
                    cur_b |= (1 << k_idx)
                    # A_k = ak ^ x_k = ak ^ (bk ^ 1).
                    # Since ak != bk, bk ^ 1 = ak. So A_k = ak ^ ak = 0.
                    # cur_a is unchanged by bit k_idx.

    return (cur_a % MOD) * (cur_b % MOD) % MOD