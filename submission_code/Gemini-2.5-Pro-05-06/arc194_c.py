import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    C = list(map(int, sys.stdin.readline().split()))
    
    cost_for_fixed_ones_sum = 0
    S1_costs = []  # Costs C_i for elements A_i=1, B_i=0 (flip 1 to 0)
    S0_costs = []  # Costs C_i for elements A_i=0, B_i=1 (flip 0 to 1)
    num_diff_elements = 0

    for i in range(N):
        if A[i] == B[i]:
            if A[i] == 1:
                cost_for_fixed_ones_sum += C[i]
        else:
            num_diff_elements += 1
            if A[i] == 1:  # Must flip from 1 to 0
                S1_costs.append(C[i])
            else:  # Must flip from 0 to 1
                S0_costs.append(C[i])

    if num_diff_elements == 0:
        print(0)
        return

    m = num_diff_elements
    total_cost = 0
    
    # Contribution from elements not flipped (A_i == B_i)
    # If A_i = 1, it remains 1 for m operations. Adds C_i each time. Total m * C_i.
    # If A_i = 0, it remains 0 for m operations. Adds 0 each time. Total 0.
    total_cost += m * cost_for_fixed_ones_sum

    # Contribution from elements flipped from 1 to 0 (S1_costs)
    # These are A_p where A_p^(0)=1. Term is C_p * (s_0(p)-1).
    # s_0(p) are ranks 1, ..., m1 (where m1 = |S1_costs|).
    # (s_0(p)-1) takes values 0, ..., m1-1.
    # To minimize sum of C_p * (s_0(p)-1), pair largest C_p with smallest (s_0(p)-1)=0, etc.
    # So, sort S1_costs in descending order. S1_costs[k] is C_p for (s_0(p)-1)=k.
    S1_costs.sort(reverse=True)
    m1 = len(S1_costs)
    for k in range(m1): # k is effectively (s_0(p)-1) for the C_p = S1_costs[k]
        total_cost += S1_costs[k] * k

    # Contribution from elements flipped from 0 to 1 (S0_costs)
    # These are A_p where A_p^(0)=0. Term is C_p * (m - s_0(p) + 1).
    # s_0(p) are ranks m1+1, ..., m.
    # Let s_0(p) = m1 + s', where s' is rank within S0 flips (1 to m0, where m0 = |S0_costs|).
    # Term becomes C_p * (m - (m1+s') + 1) = C_p * (m0 - s' + 1).
    # (m0 - s' + 1) takes values m0, m0-1, ..., 1 as s' goes from 1 to m0.
    # To minimize sum of C_p * (m0 - s' + 1), pair largest C_p with smallest factor 1, etc.
    # This means smallest C_p gets largest factor m0.
    # So, sort S0_costs in ascending order. S0_costs[k] is (k+1)-th smallest C_p, so s'=k+1.
    # Factor is (m0 - (k+1) + 1) = m0 - k.
    S0_costs.sort()
    m0 = len(S0_costs)
    for k in range(m0): 
        total_cost += S0_costs[k] * (m0 - k)
        
    print(total_cost)

solve()