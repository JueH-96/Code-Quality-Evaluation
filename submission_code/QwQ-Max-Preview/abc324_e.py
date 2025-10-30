def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    T = input[idx]
    idx += 1
    m = len(T)
    reversed_T = T[::-1]
    
    pre_list = []
    suf_list = []
    
    for _ in range(N):
        S = input[idx]
        idx += 1
        
        # Compute pre_val
        t_ptr = 0
        for c in S:
            if t_ptr < m and c == T[t_ptr]:
                t_ptr += 1
        pre_val = t_ptr
        pre_list.append(pre_val)
        
        # Compute suf_val
        t_ptr = 0
        reversed_S = reversed(S)
        for c in reversed_S:
            if t_ptr < m and c == reversed_T[t_ptr]:
                t_ptr += 1
        suf_val = t_ptr
        suf_list.append(suf_val)
    
    # Compute freq_pre
    freq_pre = [0] * (m + 1)
    for a in pre_list:
        if a > m:
            a = m
        freq_pre[a] += 1
    
    # Compute freq_suf
    freq_suf = [0] * (m + 1)
    for b in suf_list:
        if b > m:
            b = m
        freq_suf[b] += 1
    
    # Compute suffix_sum
    suffix_sum = [0] * (m + 2)
    for x in range(m, -1, -1):
        suffix_sum[x] = suffix_sum[x + 1] + freq_suf[x]
    
    # Calculate the answer
    answer = 0
    for a in range(m + 1):
        count_a = freq_pre[a]
        if count_a == 0:
            continue
        required = max(0, m - a)
        answer += count_a * suffix_sum[required]
    
    print(answer)

if __name__ == "__main__":
    main()