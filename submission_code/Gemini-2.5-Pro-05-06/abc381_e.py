import sys
import bisect

def solve():
    N, Q = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    prefix_ones = [0] * (N + 1)
    for i in range(N):
        prefix_ones[i+1] = prefix_ones[i] + (1 if S[i] == '1' else 0)

    suffix_twos = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_twos[i] = suffix_twos[i+1] + (1 if S[i] == '2' else 0)

    P_slash = []
    for i in range(N):
        if S[i] == '/':
            P_slash.append(i)
    
    K = len(P_slash)

    if K == 0:
        for _ in range(Q):
            sys.stdout.write("0
")
        return

    X_arr = [0] * K
    Y_arr = [0] * K
    D_arr = [0] * K

    for i in range(K):
        slash_idx = P_slash[i]
        X_arr[i] = prefix_ones[slash_idx] 
        Y_arr[i] = suffix_twos[slash_idx+1] 
        D_arr[i] = X_arr[i] - Y_arr[i]

    query_outputs = []
    for _ in range(Q):
        L_1idx, R_1idx = map(int, sys.stdin.readline().split())
        L0 = L_1idx - 1 
        R0 = R_1idx - 1 
        
        j_a = bisect.bisect_left(P_slash, L0)
        j_b = bisect.bisect_right(P_slash, R0) - 1

        if j_a > j_b: 
            query_outputs.append("0")
            continue

        C1 = prefix_ones[L0]
        C2 = suffix_twos[R0+1]
        
        V = C1 - C2
        
        k_s = bisect.bisect_left(D_arr, V, lo=j_a, hi=j_b + 1) 
        
        max_m = 0 

        if k_s - 1 >= j_a: 
            j = k_s - 1
            ones_count_in_T = X_arr[j] - C1
            twos_count_in_T = Y_arr[j] - C2
            max_m = max(max_m, min(ones_count_in_T, twos_count_in_T))

        if k_s <= j_b: 
            j = k_s
            ones_count_in_T = X_arr[j] - C1
            twos_count_in_T = Y_arr[j] - C2
            max_m = max(max_m, min(ones_count_in_T, twos_count_in_T))
        
        query_outputs.append(str(2 * max_m + 1))

    sys.stdout.write("
".join(query_outputs) + "
")

solve()