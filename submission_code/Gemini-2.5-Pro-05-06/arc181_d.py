import sys

def main():
    input_func = sys.stdin.readline # Cache readline

    N = int(input_func())
    P_list = list(map(int, input_func().split()))
    M = int(input_func())
    A_list = list(map(int, input_func().split()))

    # Calculate initial inversions using Fenwick tree (BIT)
    # BIT stores frequencies of values. Values are 1 to N. BIT is 1-indexed.
    bit_fen = [0] * (N + 1)
    
    def update_fen(v_idx, delta): # v_idx is a 1-based value from P_list
        while v_idx <= N:
            bit_fen[v_idx] += delta
            v_idx += v_idx & (-v_idx)

    def query_fen(v_idx): # v_idx is a 1-based value. Returns sum of frequencies for values from 1 up to v_idx.
        s = 0
        while v_idx > 0:
            s += bit_fen[v_idx]
            v_idx -= v_idx & (-v_idx)
        return s

    current_inversions = 0
    for i in range(N):
        val = P_list[i] # Current value from P_list (these are 1 to N)
        # Number of elements processed before current one = i
        # Number of elements processed before current one that are <= val = query_fen(val)
        # So, number of elements processed before current one that are > val = i - query_fen(val)
        current_inversions += i - query_fen(val)
        update_fen(val, 1)
    
    # Process operations
    # P_list is 0-indexed. Values P_list[j] are 1 to N.
    # Operation k: For i=1,2,...,k-1 (1-indexed elements P_i, P_{i+1})
    # This means 0-indexed loop for i_0 from 0 to k-2.
    # Elements involved: P_list[0]...P_list[k-1].

    results = []
    last_A_val = 0 # Using 0 as initial, since A_i ranges from 2 to N.
    streak = 0

    # current_P points to P_list, which will be modified directly.
    # No need for a separate current_P variable if P_list is used.

    for k_op_val in A_list: # k_op_val is A_s from problem, values 2 to N.
        if k_op_val == last_A_val:
            streak += 1
        else:
            last_A_val = k_op_val
            streak = 1
        
        # Number of comparisons to make in this pass: max(0, k_op_val - streak).
        # Loop for 0-indexed i from 0 up to (num_comparisons - 1).
        num_comparisons = max(0, k_op_val - streak)
        
        swaps_this_step = 0
        for i in range(num_comparisons):
            # P_list[i] vs P_list[i+1]
            if P_list[i] > P_list[i+1]:
                P_list[i], P_list[i+1] = P_list[i+1], P_list[i]
                swaps_this_step += 1
        
        current_inversions -= swaps_this_step
        results.append(str(current_inversions))

    sys.stdout.write("
".join(results) + "
")

if __name__ == '__main__':
    main()