import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    # dp[i] is a set of masks.
    # dp[i] contains 'mask' if S[0...i-1] can be formed, and 'mask' (length M)
    # describes coverage for S[i...i+M-1].
    # mask's j-th bit is 1 if S[i+j] is covered by a T placement whose
    # primary purpose was to help form S[0...i-1].
    dp = [set() for _ in range(N + 1)]
    dp[0].add(0) # Base case: empty prefix, no coverage needed for S[0...M-1] initially.
                 # Or rather, mask describes S[0...M-1] if i=0. All must be covered.
                 # Let mask bit j mean S[i+j] is covered.

    for i in range(N):
        if not dp[i]: # If S[0...i-1] cannot be formed in any way
            continue

        for prev_mask in dp[i]:
            # Option 1: S[i] is already covered by a T placement represented in prev_mask
            if (prev_mask & 1) == 1:
                dp[i+1].add(prev_mask >> 1)

            # Option 2: Place a new T to cover S[i] (or re-cover S[i])
            # This T aligns T[k_in_T] with S[i]. So it starts at op_start_idx = i - k_in_T.
            for k_in_T in range(M):
                if S[i] != T[k_in_T]:
                    continue
                
                op_start_idx = i - k_in_T
                if op_start_idx < 0:
                    continue

                # Check consistency: S[op_start_idx ... i-1] must match T[0 ... k_in_T-1]
                # This means the part of T *before* S[i] must match S.
                is_past_consistent = True
                for offset in range(k_in_T): # iterate T[0] to T[k_in_T-1]
                    s_char_idx = op_start_idx + offset
                    if s_char_idx < 0 : # Should not happen if op_start_idx >=0 and offset < k_in_T <= i-op_start_idx
                         is_past_consistent = False; break 
                    if S[s_char_idx] != T[offset]:
                        is_past_consistent = False
                        break
                if not is_past_consistent:
                    continue
                
                # If placement is valid, calculate its effect on the next state's mask.
                # The next state is for dp[i+1]. Its mask describes S[i+1 ... i+M].
                # This new T covers S[op_start_idx ... op_start_idx+M-1].
                # Its contribution to the mask for S[i+1 ... i+M] is:
                current_op_effect_mask = 0
                for j in range(M - 1): # Corresponds to S[i+1+j], which is mask index j for dp[i+1]
                    # Character in T for S[i+1+j] is T[k_in_T + 1 + j]
                    t_char_idx_for_s_i_plus_1_plus_j = k_in_T + 1 + j
                    
                    if t_char_idx_for_s_i_plus_1_plus_j < M: # Current T op covers this far
                        s_idx_to_check = i + 1 + j
                        if s_idx_to_check < N: # And it's within S bounds
                            if S[s_idx_to_check] == T[t_char_idx_for_s_i_plus_1_plus_j]:
                                current_op_effect_mask |= (1 << j)
                        # else (s_idx_to_check >= N): this part of mask is for beyond S, so effectively 0
                    # else: current T op does not extend this far.
                
                # The resulting mask for dp[i+1] is the union of what prev_mask implies
                # and what the current_op implies for S[i+1 ... i+M].
                # If S[i] was covered by prev_mask (prev_mask&1 == 1), then prev_mask>>1 contributes.
                # If S[i] was NOT, then prev_mask>>1 doesn't provide coverage to S[i+1...] from *that specific path*.
                # It's simpler: the next mask is (prev_mask >> 1 if S[i] was covered by it) OR current_op_effect_mask.
                # No, this is wrong. A position is covered if *any* T covers it.
                # The bits of `prev_mask >> 1` represent carry-over coverage.
                # `current_op_effect_mask` represents coverage from this new T placement.
                # These are alternative ways to achieve coverage.
                # So, dp[i+1] gets (prev_mask >> 1) ORed with current_op_effect_mask.
                # This ORing happens implicitly because we add to a set.
                # But the mask passed should be formed by one path.
                # If S[i] is covered by prev_mask&1, its shifted version is a candidate.
                # Separately, placing T now provides coverage described by current_op_effect_mask. This is another candidate.
                # No, current_op_effect_mask is just for the newly placed T. The mask for dp[i+1] needs to inherit.
                # The mask describes the state of S[i+1...i+M].
                # If S[i] was covered by (prev_mask & 1), then dp[i+1] gets (prev_mask >> 1).
                # If we additionally place T now, this creates a *new* state for dp[i+1].
                # The mask for this new state is current_op_effect_mask. (No, this is also wrong.)
                
                # Correct logic for mask propagation:
                # The mask `m` for `dp[i+1]` describes coverage for `S[i+1...i+M]`.
                # Bit `j` of `m` corresponds to `S[i+1+j]`.
                # `S[i+1+j]` can be covered EITHER by `prev_mask` (if `prev_mask` bit `j+1` was set)
                # OR by the new T operation (if `T[k_in_T+1+j] == S[i+1+j]`).
                # So, `m_bit_j = (prev_mask_bit_{j+1}) OR (new_T_op_covers_it_and_matches)`.
                
                # Let next_actual_mask be the mask for dp[i+1]
                next_actual_mask = 0
                for j_mask_idx in range(M): # for S[i+1 ... i+1+M-1]
                    s_char_real_idx = i + 1 + j_mask_idx
                    if s_char_real_idx >= N: break # beyond S

                    covered_by_prev = (prev_mask >> (j_mask_idx + 1)) & 1
                    
                    covered_by_new_op = 0
                    # Character in T for S[s_char_real_idx] is T[s_char_real_idx - op_start_idx]
                    idx_in_T_for_new_op = s_char_real_idx - op_start_idx
                    if 0 <= idx_in_T_for_new_op < M: # Covered by new op
                        if S[s_char_real_idx] == T[idx_in_T_for_new_op]:
                            covered_by_new_op = 1
                    
                    if covered_by_prev or covered_by_new_op:
                        next_actual_mask |= (1 << j_mask_idx)
                
                dp[i+1].add(next_actual_mask)

    # Check final state: dp[N] must contain some mask. Any mask is fine, typically 0.
    if dp[N]: # or specifically check for dp[N].contains(0) if mask means "debt"
        print("Yes")
    else:
        print("No")

solve()