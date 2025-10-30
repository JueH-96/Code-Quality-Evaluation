# YOUR CODE HERE
import sys
import numpy as np
from collections import defaultdict

def main():
    import sys
    import sys
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    # Step 1: Compute k_i and u_i for each A_i
    groups = defaultdict(list)
    sum_u = 0
    for a in A:
        k = 0
        while a %2 ==0:
            k +=1
            a = a //2
        u = a
        groups[k].append(u)
        sum_u +=u
    
    # Get all unique k's sorted
    unique_ks = sorted(groups.keys())
    max_k = unique_ks[-1] if unique_ks else 0
    
    # Step 3: Precompute sum_c_u and sum_c_cnt for each k
    sum_c_u = {}
    sum_c_cnt = {}
    for k in unique_ks:
        u_list = groups[k]
        sum_c_u[k] = sum(u_list)
        sum_c_cnt[k] = len(u_list)
    
    # Step 5: Compute sum over k_i <k_j
    sum_total =0
    for i in range(len(unique_ks)):
        k_i = unique_ks[i]
        for j in range(i+1, len(unique_ks)):
            k_j = unique_ks[j]
            delta =k_j -k_i
            pow2 =1 <<delta
            sum_i_u = sum_c_u[k_i]
            sum_i_cnt = sum_c_cnt[k_i]
            sum_j_u = sum_c_u[k_j]
            sum_j_cnt = sum_c_cnt[k_j]
            contribution = sum_i_u * sum_j_cnt + pow2 * sum_j_u * sum_i_cnt
            sum_total += contribution
    
    # Step 6: Compute sum over k_i =k_j
    for k in unique_ks:
        u_list = groups[k]
        m = len(u_list)
        if m <2:
            continue
        u = np.array(u_list, dtype=np.int64)
        # Compute all possible u_i + u_j
        x = u[:, np.newaxis] + u[np.newaxis, :]  # m x m matrix
        # Compute f(x) = x // (x & -x)
        f_x = x // (x & -x)
        sum_all = f_x.sum()
        # Compute sum of f(2*u_i) which is u_i
        sum_self = ( (2*u) // ( (2*u) & -(2*u) ) ).sum()
        # Sum for i < j
        sum_i_less_j = (sum_all - sum_self) //2
        sum_total +=sum_i_less_j
    
    # Step 7: Add sum_u
    sum_total +=sum_u
    
    print(sum_total)

if __name__ == "__main__":
    main()