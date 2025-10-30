def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    
    # Compute prefix sums S_i modulo M for i from 0 to N-1
    S = [0] * N
    for i in range(N):
        S[i] = (S[i-1] + A[i]) % M if i > 0 else A[i] % M
    
    # sum_A_mod is the total sum of A_i modulo M
    sum_A_mod = sum(A) % M
    
    # Initialize freq_future with frequencies of S_{t-1} for t from 1 to N
    freq_future = [0] * M
    for t in range(1, N+1):
        s_t_minus_1 = S[t-1] if t-1 >= 0 else 0
        freq_future[s_t_minus_1 % M] += 1
    
    # Initialize freq_past with frequencies of S_{t-1} for t from 0 to 0 (initially empty)
    freq_past = [0] * M
    
    answer = 0
    
    # Iterate s from 1 to N
    for s in range(1, N+1):
        s_minus_1 = s - 1
        # Case 1: t > s
        if s_minus_1 >= 0:
            x = S[s_minus_1] % M
            answer += freq_future[x]
        
        # Case 2: t <= s
        if s_minus_1 >= 0:
            x = (S[s_minus_1] - sum_A_mod) % M
            answer += freq_past[x]
        
        # Update freq_future: remove S_s modulo M
        if s < N:
            x = S[s] % M
            freq_future[x] -= 1
        
        # Update freq_past: add S_{s-1} modulo M
        if s_minus_1 >= 0:
            x = S[s_minus_1] % M
            freq_past[x] += 1
    
    print(answer)

if __name__ == '__main__':
    main()