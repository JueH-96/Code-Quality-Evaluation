MOD = 998244353

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()
    
    def get_new_state(s, char):
        a = (s >> 2) & 1
        b = (s >> 1) & 1
        c_parity = s & 1
        new_a = (a + (char == 'A')) % 2
        new_b = (b + (char == 'B')) % 2
        new_c = (c_parity + (char == 'C')) % 2
        return new_a * 4 + new_b * 2 + new_c
    
    # Initialize DP and frequency
    curr_dp = [[0] * (K + 1) for _ in range(8)]
    curr_dp[0][0] = 1
    freq_prev = [0] * 8
    freq_prev[0] = 1
    
    for i in range(N):
        c = S[i]
        allowed = []
        if c == 'A':
            allowed = ['A']
        elif c == 'B':
            allowed = ['B']
        elif c == 'C':
            allowed = ['C']
        else:
            allowed = ['A', 'B', 'C']
        
        next_dp = [[0] * (K + 1) for _ in range(8)]
        
        for s_prev in range(8):
            for cnt_prev in range(K + 1):
                if curr_dp[s_prev][cnt_prev] == 0:
                    continue
                for char in allowed:
                    s_new = get_new_state(s_prev, char)
                    delta = freq_prev[s_new] + freq_prev[s_new ^ 7]
                    new_cnt = cnt_prev + delta
                    if new_cnt > K:
                        new_cnt = K
                    next_dp[s_new][new_cnt] = (next_dp[s_new][new_cnt] + curr_dp[s_prev][cnt_prev]) % MOD
        
        # Compute new frequency
        freq_next = [0] * 8
        for s in range(8):
            for cnt in range(K + 1):
                freq_next[s] = (freq_next[s] + next_dp[s][cnt]) % MOD
        
        curr_dp = next_dp
        freq_prev = freq_next
    
    total = 0
    for s in range(8):
        for cnt in range(K, K + 1):
            if cnt <= K:
                total = (total + curr_dp[s][cnt]) % MOD
    print(total)

if __name__ == '__main__':
    main()