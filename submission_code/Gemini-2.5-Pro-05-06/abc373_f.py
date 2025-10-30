import sys
from collections import deque # Use collections.deque for efficiency

def solve():
    N, W = map(int, sys.stdin.readline().split())
    items_data = []
    for _ in range(N):
        w, v = map(int, sys.stdin.readline().split())
        items_data.append({'w': w, 'v': v})

    dp = [-1] * (W + 1)
    dp[0] = 0
    
    # dp_temp is used to store results for the current item type (dp_new in logic)
    # while dp holds results from previous item types (dp_old in logic)
    # At the end of processing an item type, dp will be updated.

    for item_idx in range(N):
        w_i = items_data[item_idx]['w']
        v_i = items_data[item_idx]['v']

        # Max number of items of type i to consider: k_i <= M_i
        # M_i = min(floor(W / w_i), v_i)
        # if v_i is 0, M_i will be 0. if w_i > W, floor(W/w_i) is 0, so M_i is 0.
        if v_i == 0 : # Optimization: if v_i is 0, H_i(k_i) = -k_i^2. Max is 0 for k_i=0. No items taken.
            continue
        
        M_i = W // w_i 
        if v_i < M_i : # if v_i is smaller, it's the true limit for non-negative happiness
             M_i = v_i
        
        if M_i == 0: # Cannot take any item of this type (e.g. w_i > W and M_i based on W//w_i became 0)
            continue
        
        # dp_next_iter will become the new dp table
        # Initialize with current dp values (equivalent to taking k=0 items of current type)
        dp_next_iter = list(dp)

        for r in range(w_i):
            # Sliding window maximum / CHT deque
            # Deque stores 'p' values. Line L_p(x) = (v_i + 2*p)*x + C_p
            # C_p = dp[p*w_i+r] - p*v_i - p*p
            # We query L_p(q) and subtract q*q
            dq = deque()
            
            max_q_for_rem = (W - r) // w_i
            for q in range(max_q_for_rem + 1):
                current_j = q * w_i + r

                # --- Query Part (find best k >= 1) ---
                # Remove p from front if k = q - p > M_i  => p < q - M_i
                if dq and dq[0] < q - M_i:
                    dq.popleft()

                if dq:
                    # Query: Slopes M_p = v_i + 2*p are increasing. Query points q are increasing.
                    # Pop from front of deque while L_front is worse than L_second_front at x=q
                    while len(dq) >= 2:
                        p_a = dq[0]
                        p_b = dq[1]
                        
                        # Value from L_pa(q) - q*q vs L_pb(q) - q*q
                        # Effectively comparing L_pa(q) vs L_pb(q)
                        val_a = (v_i + 2*p_a)*q + (dp[p_a*w_i+r] - p_a*v_i - p_a*p_a)
                        val_b = (v_i + 2*p_b)*q + (dp[p_b*w_i+r] - p_b*v_i - p_b*p_b)

                        if val_a <= val_b: # If L_pa is worse or equal, pop it
                            dq.popleft()
                        else:
                            break
                    
                    p_star = dq[0]
                    # value_from_k_items = dp[p_star*w_i+r] + H_i(k=q-p_star)
                    # This is (dp[p_star*w_i+r] - p_star*v_i - p_star*p_star) + (v_i + 2*p_star)*q - q*q
                    max_happiness_contrib = (dp[p_star*w_i+r] - p_star*v_i - p_star*p_star) + \
                                            (v_i + 2*p_star)*q - q*q
                    
                    if dp_next_iter[current_j] == -1 or max_happiness_contrib > dp_next_iter[current_j]:
                        dp_next_iter[current_j] = max_happiness_contrib
                
                # --- Addition Part (add line for p_new = q) ---
                # This line L_q will be used for future states q_future > q,
                # representing dp_old[q*w_i+r] as a base for taking (q_future-q) items.
                if dp[current_j] != -1: # If current_j is a reachable state from previous items
                    p_new = q 
                    
                    const_term_p_new = dp[current_j] - p_new*v_i - p_new*p_new
                    slope_p_new = v_i + 2*p_new

                    # Maintain CHT (upper envelope of lines). Slopes M_p are increasing.
                    # Pop from back if L_new makes L_back redundant with L_back2.
                    # Condition for L2 (p_2) to be popped: x_12 >= x_23, where x_ab is intersection of L_a and L_b.
                    # (C2-C1)(M2-M3) >= (C3-C2)(M1-M2)
                    while len(dq) >= 2:
                        p_1 = dq[-2] 
                        p_2 = dq[-1] 
                        
                        const_term_p_1 = dp[p_1*w_i+r] - p_1*v_i - p_1*p_1
                        slope_p_1 = v_i + 2*p_1
                        const_term_p_2 = dp[p_2*w_i+r] - p_2*v_i - p_2*p_2
                        slope_p_2 = v_i + 2*p_2
                        
                        # Check: (c2-c1)(m2-m3) >= (c3-c2)(m1-m2)
                        # Using (slope_p_new - slope_p_2) instead of (slope_p_2 - slope_p_new)
                        # and (slope_p_2 - slope_p_1) instead of (slope_p_1 - slope_p_2)
                        # This flips inequality signs if denominators are negative.
                        # Original CHT condition $(c_2-c_1)(m_2-m_3) \le (c_3-c_2)(m_1-m_2)$ to keep $L_2$.
                        # Pop $L_2$ if $(c_2-c_1)(m_2-m_3) > (c_3-c_2)(m_1-m_2)$.
                        # Numerators: (const_term_p_2 - const_term_p_1) vs (const_term_p_new - const_term_p_2)
                        # Denominators: (slope_p_1 - slope_p_2) vs (slope_p_2 - slope_p_new)
                        # Both denominators are negative.
                        # Pop if (C2-C1)/(M1-M2) >= (C_NEW-C2)/(M2-M_NEW)
                        # (C2-C1)*(M2-M_NEW) <= (C_NEW-C2)*(M1-M2) (denominators negative, product positive)
                        # This is (C2-C1)*(M_NEW-M2) >= (C_NEW-C2)*(M2-M1)
                        if (const_term_p_2 - const_term_p_1) * (slope_p_new - slope_p_2) >= \
                           (const_term_p_new - const_term_p_2) * (slope_p_2 - slope_p_1):
                            dq.pop()
                        else:
                            break
                    dq.append(p_new)
        dp = dp_next_iter


    max_total_happiness = 0
    for happiness_val in dp:
        if happiness_val != -1: 
            max_total_happiness = max(max_total_happiness, happiness_val)
            
    print(max_total_happiness)

solve()