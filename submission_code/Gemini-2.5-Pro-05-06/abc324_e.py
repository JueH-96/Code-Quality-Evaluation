import sys

def main():
    N_str, T_str = sys.stdin.readline().split()
    N = int(N_str)
    
    S_list = [sys.stdin.readline().strip() for _ in range(N)]

    M = len(T_str)

    prefs = [0] * N
    suffs = [0] * N

    # Calculate prefixes for each S_i
    # prefs[i] = length of the longest prefix of T_str that is a subsequence of S_list[i]
    for i in range(N):
        s_k = S_list[i]
        
        t_ptr = 0
        for char_s in s_k: # Iterate directly over characters of S_k
            if t_ptr < M and char_s == T_str[t_ptr]:
                t_ptr += 1
            # Optimization: if all of T_str is matched, no need to check further
            if t_ptr == M:
                break
        prefs[i] = t_ptr

    # Calculate suffixes for each S_i
    # suffs[i] = length of the longest suffix of T_str that is a subsequence of S_list[i]
    for i in range(N):
        s_k = S_list[i]
        
        t_ptr = M - 1 # Points to the last char of T_str initially
        s_len_matched = 0
        # Iterate S_k from right to left
        for char_s in reversed(s_k): # Iterate over characters of S_k in reverse
            if t_ptr >= 0 and char_s == T_str[t_ptr]:
                t_ptr -= 1
                s_len_matched += 1
            # Optimization: if all of T_str is matched (as a suffix), no need to check further
            if s_len_matched == M:
                break
        suffs[i] = s_len_matched
    
    # Count pairs (i, j) such that prefs[i] + suffs[j] >= M
    
    # freq_suff[k] stores the count of strings S_j such that suffs[j] == k.
    # Array size is M+1 to store counts for lengths 0 to M.
    freq_suff = [0] * (M + 1)
    for val_suff in suffs:
        freq_suff[val_suff] += 1
    
    # count_ge_suff[k] stores the count of strings S_j such that suffs[j] >= k.
    # count_ge_suff[k] = sum(freq_suff[x] for x from k to M).
    # Array size is M+1 for k from 0 to M.
    count_ge_suff = [0] * (M + 1)
    
    # This calculation is correct for M >= 0.
    # Since problem constraints state |T| >= 1, M >= 1.
    count_ge_suff[M] = freq_suff[M]
    for k in range(M - 1, -1, -1):
        count_ge_suff[k] = count_ge_suff[k+1] + freq_suff[k]
    
    total_count = 0
    for val_pref in prefs:
        # We need S_j such that suffs[j] >= M - val_pref.
        # The number of such S_j is count_ge_suff[target_suff_val].
        target_suff_val = M - val_pref
        
        # Since val_pref is in [0, M], M - val_pref is also in [0, M].
        # So target_suff_val is always a valid index for count_ge_suff.
        total_count += count_ge_suff[target_suff_val]
        
    print(total_count)

if __name__ == '__main__':
    main()