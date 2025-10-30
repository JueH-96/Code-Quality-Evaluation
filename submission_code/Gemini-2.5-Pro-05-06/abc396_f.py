import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    A_list = list(map(int, sys.stdin.readline().split()))

    sums_2_1_indexed_pos = [0] * M 
    counts = [0] * M
    for i in range(N): # 0-indexed i
        val = A_list[i]
        sums_2_1_indexed_pos[val] += 2 * (i + 1) # (i+1) is 1-indexed position
        counts[val] += 1

    bit_array = [0] * (M + 1)

    def update_bit(val, delta): # val is 0-indexed
        internal_idx = val + 1 
        while internal_idx <= M:
            bit_array[internal_idx] += delta
            internal_idx += internal_idx & (-internal_idx)

    def query_bit_sum_less_than(query_val_exclusive): # query_val_exclusive is 0-indexed
        if query_val_exclusive <= 0:
            return 0
        internal_idx_upper = query_val_exclusive
        s = 0
        while internal_idx_upper > 0:
            s += bit_array[internal_idx_upper]
            internal_idx_upper -= internal_idx_upper & (-internal_idx_upper)
        return s

    inv0 = 0
    for i in range(N - 1, -1, -1):
        val_at_A_i = A_list[i]
        inv0 += query_bit_sum_less_than(val_at_A_i)
        update_bit(val_at_A_i, 1)

    ans = [0] * M
    ans[0] = inv0
    current_inv = inv0

    for k_loop in range(M - 1): 
        original_A_val_whose_B_val_was_M_minus_1 = (M - 1 - k_loop) 
        
        c_k = counts[original_A_val_whose_B_val_was_M_minus_1]
        sum_2p_k = sums_2_1_indexed_pos[original_A_val_whose_B_val_was_M_minus_1]

        if c_k > 0:
            delta = sum_2p_k - c_k * (N + 1)
            current_inv += delta
        
        ans[k_loop + 1] = current_inv
        
    sys.stdout.write('
'.join(map(str, ans)) + '
')

solve()