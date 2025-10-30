import sys

INF = float('inf')

# Memoization caches, cleared for each DP step transition (q_idx)
# This is because the set of (l1,r1,l2,r2) pairs encountered might be sparse
# Global memoization could lead to very large cache.
# However, for dist_one and calculate_moves, N is fixed.
# If N is small, global memo can be good. If N is large, local (per-q_idx) is safer.
# For QN^2, we need O(1) cost for calculate_moves.
# dist_one has N^3 states. calculate_moves has N^4 states.
# Pre-calculating all N^4 states for calculate_moves is too much memory.
# The functions themselves are O(1) if not memoized.
# Let's remove per-step memoization clearing and rely on Python's dicts for performance.
# Or, make them non-memoized O(1) calculations.

# Global N for helper functions
_N_GLOBAL = 0

def ring_dist_func(p1, p2):
    # Assumes _N_GLOBAL is set
    if p1 == p2:
        return 0
    p1_0, p2_0 = p1 - 1, p2 - 1 # 0-indexed
    diff = abs(p1_0 - p2_0)
    return min(diff, _N_GLOBAL - diff)

def dist_one_func(curr_p, targ_p, fixed_p):
    # Assumes _N_GLOBAL is set
    if curr_p == targ_p:
        return 0
    if targ_p == fixed_p: # Cannot move to where the other hand is
        return INF

    curr_p0, targ_p0, fixed_p0 = curr_p - 1, targ_p - 1, fixed_p - 1 # 0-indexed

    cost_cw = (targ_p0 - curr_p0 + _N_GLOBAL) % _N_GLOBAL
    cost_ccw = (curr_p0 - targ_p0 + _N_GLOBAL) % _N_GLOBAL
    
    blocked_cw = False
    if cost_cw > 0: # Path has intermediate points
        # Is fixed_p0 strictly between curr_p0 and targ_p0 on CW path?
        if (fixed_p0 - curr_p0 + _N_GLOBAL) % _N_GLOBAL < cost_cw:
            blocked_cw = True
    
    blocked_ccw = False
    if cost_ccw > 0: # Path has intermediate points
        # Is fixed_p0 strictly between curr_p0 and targ_p0 on CCW path?
        if (curr_p0 - fixed_p0 + _N_GLOBAL) % _N_GLOBAL < cost_ccw:
            blocked_ccw = True
    
    res = INF
    if not blocked_cw:
        res = min(res, cost_cw)
    if not blocked_ccw:
        res = min(res, cost_ccw)
    
    return res

memo_calculate_moves = {}
def calculate_moves_func(l1, r1, l2, r2):
    # Assumes _N_GLOBAL is set
    state_key = (l1, r1, l2, r2, _N_GLOBAL) # Include N in key if it could change (not here)
    if state_key in memo_calculate_moves:
        return memo_calculate_moves[state_key]

    if l1 == l2 and r1 == r2:
        return 0
    
    # Basic check for invalid states, though DP logic should prevent l1=r1 or l2=r2.
    # if l1 == r1 or l2 == r2: return INF 

    # Case 1: Only L moves
    if r1 == r2:
        res = dist_one_func(l1, l2, r1)
        memo_calculate_moves[state_key] = res
        return res
        
    # Case 2: Only R moves
    if l1 == l2:
        res = dist_one_func(r1, r2, l1)
        memo_calculate_moves[state_key] = res
        return res

    # Case 3: Swap positions (l2=r1, r2=l1)
    if l1 == r2 and r1 == l2:
        # Cost is sum of distances each hand travels, plus 1 if they were adjacent.
        # dist L moves: ring_dist_func(l1, l2) which is ring_dist_func(l1, r1)
        # dist R moves: ring_dist_func(r1, r2) which is ring_dist_func(r1, l1)
        # Total: 2 * ring_dist_func(l1, r1)
        # If ring_dist_func(l1, r1) == 1 (adjacent), add 1.
        dist_between_hands = ring_dist_func(l1, r1)
        res = 2 * dist_between_hands
        if dist_between_hands == 1:
             res += 1
        memo_calculate_moves[state_key] = res
        return res

    # Case 4: General - L moves then R moves, or R moves then L moves
    # L then R:
    cost1 = INF
    moves_l_first = dist_one_func(l1, l2, r1)
    if moves_l_first != INF:
        moves_r_second = dist_one_func(r1, r2, l2)
        if moves_r_second != INF:
            cost1 = moves_l_first + moves_r_second
    
    # R then L:
    cost2 = INF
    moves_r_first = dist_one_func(r1, r2, l1)
    if moves_r_first != INF:
        moves_l_second = dist_one_func(l1, l2, r2)
        if moves_l_second != INF:
            cost2 = moves_r_first + moves_l_second
    
    res = min(cost1, cost2)
    memo_calculate_moves[state_key] = res
    return res

def solve():
    global _N_GLOBAL
    N_val, Q_val = map(int, sys.stdin.readline().split())
    _N_GLOBAL = N_val # Set global N

    queries = []
    for _ in range(Q_val):
        parts = sys.stdin.readline().split()
        queries.append((parts[0], int(parts[1])))
    
    dp_prev = [INF] * (N_val + 1)
    current_l, current_r = 1, 2 # Initial positions

    for i in range(Q_val):
        H_i, T_i = queries[i]
        
        # Clear memoization specific to calculate_moves for this N, if it could change often.
        # For fixed N, it's okay to keep it global.
        # If the number of distinct (l1,r1,l2,r2) states accessed is small, memo helps.
        # Max N^4 states. With N=3000, this is too large.
        # However, the number of distinct (l1,r1) start states is N, and (l2,r2) end states is N.
        # So overall distinct calls to calculate_moves is N*N per DP step if not memoized.
        # Given QN^2 structure, it is better to make calculate_moves O(1) (not memoized).
        # The dict lookups for memoization may add overhead.
        # For this version, let's assume calculate_moves_func itself is fast enough (O(1))
        # and use memoization for its potentially repeated internal calls.
        # A better approach for QN^2: make calculate_moves_func non-memoized.
        # It calls dist_one_func (non-memoized, O(1)). So calculate_moves is O(1).

        # Let's use memoization for calculate_moves_func, but not clear it.
        # It caches results for (l1,r1,l2,r2) tuples. Total states could be up to N^4.
        # However, specific (l1,r1) comes from (T_{prev},k) or (k,T_{prev}), and (l2,r2) from (T_{curr},j) or (j,T_{curr}).
        # So only O(N^2) distinct calls to calculate_moves_func per DP step.
        # The number of values for T_prev, T_curr is small (at most Q).
        # This means total distinct calls might be Q*N^2, not N^4.
        # Python dicts are efficient; let's keep the memoization for calculate_moves_func.

        dp_curr = [INF] * (N_val + 1)

        if i == 0:
            for other_hand_final_pos_j in range(1, N_val + 1):
                if H_i == 'L':
                    target_l, target_r = T_i, other_hand_final_pos_j
                else: 
                    target_l, target_r = other_hand_final_pos_j, T_i

                if target_l == target_r: continue

                ops_this_step = calculate_moves_func(current_l, current_r, target_l, target_r)
                if ops_this_step != INF:
                    dp_curr[other_hand_final_pos_j] = ops_this_step
        else:
            H_prev, T_prev = queries[i-1]
            for other_hand_final_pos_j in range(1, N_val + 1):
                if H_i == 'L':
                    target_l, target_r = T_i, other_hand_final_pos_j
                else: 
                    target_l, target_r = other_hand_final_pos_j, T_i

                if target_l == target_r: continue
                
                current_min_ops = INF
                for prev_other_hand_pos_k in range(1, N_val + 1):
                    if dp_prev[prev_other_hand_pos_k] == INF: continue

                    if H_prev == 'L':
                        l_p, r_p = T_prev, prev_other_hand_pos_k
                    else: 
                        l_p, r_p = prev_other_hand_pos_k, T_prev
                    
                    # This check should ideally not be needed if dp_prev states are valid
                    if l_p == r_p : continue 

                    ops_this_step = calculate_moves_func(l_p, r_p, target_l, target_r)
                    
                    if ops_this_step != INF:
                        current_min_ops = min(current_min_ops, dp_prev[prev_other_hand_pos_k] + ops_this_step)
                
                dp_curr[other_hand_final_pos_j] = current_min_ops
        
        dp_prev = dp_curr

    min_total_ops_final = INF
    for ops in dp_prev:
        min_total_ops_final = min(min_total_ops_final, ops)
    
    # If min_total_ops_final is still INF, it means no path was found (should not happen per problem statement)
    print(min_total_ops_final if min_total_ops_final != INF else -1) # Or handle as error

solve()