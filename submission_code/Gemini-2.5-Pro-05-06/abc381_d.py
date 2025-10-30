import sys

def main():
    N = int(sys.stdin.readline())
    if N == 0:
        print(0)
        return
    A = list(map(int, sys.stdin.readline().split()))

    if N < 2: # Cannot form any pair
        print(0)
        return

    # dp[i] stores the length of the 1122-sequence ending at index i (A[i])
    dp = [0] * N 
    
    # last_idx_map[val] stores the 0-indexed index `k` such that A[k] = val,
    # and A[k] was the first element of a pair (v_j, v_j) in a previously computed
    # 1122-sequence. When we consider the pair (A[i-1], A[i]), A[i-1] is the
    # current v_k, so last_idx_map[val] stores the index of A[i-1] for this context.
    last_idx_map = {} 

    max_len_overall = 0

    # dp[0] is 0 (no A[-1] to form a pair with A[0]).
    # Loop for i from 1 to N-1.
    for i in range(1, N):
        # A 1122-sequence must end with a pair (X,X).
        # So, if A[i] != A[i-1], no 1122-sequence can end at A[i].
        if A[i] != A[i-1]:
            dp[i] = 0
            continue # last_idx_map is not updated with A[i-1] from this position.

        val = A[i] # Current value for the pair (val, val)

        # Length of the 1122-sequence ending at A[i-2].
        # If i < 2 (i.e., i=1), there's no A[i-2], so previous sequence length is 0.
        len_prev_seq = dp[i-2] if i >= 2 else 0
        
        # Candidate length by extending the previous sequence: dp[i-2] + 2
        candidate_len_by_extension = len_prev_seq + 2

        # The list of v_j's for the sequence ending at A[i-2] (of length len_prev_seq)
        # starts with A[s'], where s' is the index of its first v_j.
        # s' = (i-2) - len_prev_seq + 1.
        # This can be rewritten relative to A[i-1] (the current v_k).
        # The first v_j in the list v_1,...,v_{k-1} is A[(i-1) - len_prev_seq].
        # Let this index be first_v_idx_in_chain.
        first_v_idx_in_chain = (i-1) - len_prev_seq
        
        # Get the index where `val` previously occurred as a v_j.
        # Defaults to -1 if not found or if map is empty.
        prev_occurrence_of_val_idx = last_idx_map.get(val, -1)

        if prev_occurrence_of_val_idx < first_v_idx_in_chain:
            # The previous occurrence of `val` (if any) is "outside" the active chain of
            # v_j's (v_1, ..., v_{k-1}) forming the sequence for dp[i-2].
            # So, `val` is distinct from these v_j's. We can extend.
            dp[i] = candidate_len_by_extension
        else:
            # `val` collides with some v_j in the active chain.
            # The new sequence is limited by this collision. It must start after
            # the pair involving A[prev_occurrence_of_val_idx].
            # The v_j's are A[prev_occurrence_of_val_idx+2], A[prev_occurrence_of_val_idx+4], ..., A[i-1].
            # The length is (i-1) - prev_occurrence_of_val_idx.
            dp[i] = (i-1) - prev_occurrence_of_val_idx
        
        # Update last_idx_map: A[i-1] (which is `val`) is now the latest v_j encountered with this value.
        last_idx_map[val] = i-1 
        
        if dp[i] > max_len_overall:
            max_len_overall = dp[i]
            
    print(max_len_overall)

if __name__ == '__main__':
    main()