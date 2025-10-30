import sys
from collections import deque

def main():
    data = sys.stdin.readline().split()
    N = int(data[0])
    P = int(data[1])
    MOD = P
    MAX_EDGES = N * (N - 1) // 2
    
    max_n_cr = 1000
    nCr_table = [[0] * (max_n_cr + 1) for _ in range(max_n_cr + 1)]
    for n in range(0, max_n_cr + 1):
        for r in range(0, n + 1):
            if r == 0 or r == n:
                nCr_table[n][r] = 1 % MOD
            else:
                nCr_table[n][r] = (nCr_table[n - 1][r - 1] + nCr_table[n - 1][r]) % MOD

    T_array = [[None] * (N + 1) for _ in range(N + 1)]
    for a_val in range(0, N + 1):
        for s_val in range(0, N + 1):
            L_between = a_val * s_val
            poly_between = []
            if L_between < 0:
                poly_between = [1]
            else:
                poly_between = [0] * (L_between + 1)
                for k in range(0, L_between + 1):
                    cover = 0
                    for j in range(0, s_val + 1):
                        rem = s_val - j
                        avail_edges = a_val * rem
                        if k > avail_edges:
                            term = 0
                        else:
                            term = nCr_table[s_val][j] * nCr_table[avail_edges][k] % MOD
                        if j % 2 == 1:
                            term = (-term) % MOD
                        cover = (cover + term) % MOD
                    poly_between[k] = cover
            
            within_edges = s_val * (s_val - 1) // 2
            if within_edges < 0:
                poly_within = [1]
            else:
                poly_within = [0] * (within_edges + 1)
                for k in range(0, within_edges + 1):
                    poly_within[k] = nCr_table[within_edges][k] % MOD
            
            len1 = len(poly_within)
            len2 = len(poly_between)
            T_len = len1 + len2 - 1
            T_arr_here = [0] * T_len
            for i in range(len1):
                for j in range(len2):
                    idx = i + j
                    if idx < T_len:
                        T_arr_here[idx] = (T_arr_here[idx] + poly_within[i] * poly_between[j]) % MOD
            T_array[a_val][s_val] = T_arr_here

    dp_dict = {}
    start_state = (0, 1, 1, 0)
    vec_start = [0] * (MAX_EDGES + 1)
    vec_start[0] = 1
    dp_dict[start_state] = vec_start
    queue = deque([start_state])
    
    while queue:
        state = queue.popleft()
        i, a, e, o = state
        vec = dp_dict[state]
        R = N - (e + o)
        if R == 0:
            continue
        for s in range(1, R + 1):
            if (i + 1) % 2 == 0:
                new_e = e + s
                new_o = o
            else:
                new_e = e
                new_o = o + s
            if new_e > N // 2 or new_o > N // 2:
                continue
            new_state = (i + 1, s, new_e, new_o)
            comb_vertices = nCr_table[R][s]
            T_arr_here = T_array[a][s]
            L_T = len(T_arr_here)
            new_vec = [0] * (MAX_EDGES + 1)
            for t_current in range(0, MAX_EDGES + 1):
                if vec[t_current] == 0:
                    continue
                max_k = min(L_T, MAX_EDGES - t_current + 1)
                for k in range(0, max_k):
                    if T_arr_here[k] != 0:
                        total_edges = t_current + k
                        if total_edges <= MAX_EDGES:
                            new_vec[total_edges] = (new_vec[total_edges] + vec[t_current] * T_arr_here[k]) % MOD
            for idx in range(0, MAX_EDGES + 1):
                new_vec[idx] = new_vec[idx] * comb_vertices % MOD
            if new_state not in dp_dict:
                dp_dict[new_state] = [0] * (MAX_EDGES + 1)
                queue.append(new_state)
            current_vec = dp_dict[new_state]
            for idx in range(0, MAX_EDGES + 1):
                current_vec[idx] = (current_vec[idx] + new_vec[idx]) % MOD
    result_arr = [0] * (MAX_EDGES + 1)
    for state, vec in dp_dict.items():
        i_val, a_val, e_val, o_val = state
        if e_val == N // 2 and o_val == N // 2:
            for t in range(0, MAX_EDGES + 1):
                result_arr[t] = (result_arr[t] + vec[t]) % MOD
    output_list = []
    for M in range(N - 1, MAX_EDGES + 1):
        output_list.append(str(result_arr[M]))
    print(" ".join(output_list))

if __name__ == '__main__':
    main()