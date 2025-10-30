MOD = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(0)
        return
    parts = data[0].split()
    N = int(parts[0])
    K = int(parts[1])
    S = data[1].strip()
    
    if K > N:
        total_question = S.count('?')
        total_ways = pow(2, total_question, MOD)
        print(total_ways)
        return
        
    L = K - 1
    total_states = 1 << L
    total_pal = 1 << K

    pal = [False] * total_pal
    for x in range(total_pal):
        flag = True
        for j in range(K // 2):
            left_bit = (x >> (K - 1 - j)) & 1
            right_bit = (x >> j) & 1
            if left_bit != right_bit:
                flag = False
                break
        pal[x] = flag

    dp = [0] * total_states
    if S[0] == 'A' or S[0] == '?':
        dp[0] = (dp[0] + 1) % MOD
    if S[0] == 'B' or S[0] == '?':
        dp[1] = (dp[1] + 1) % MOD

    for i in range(1, N):
        new_dp = [0] * total_states
        for state_old in range(total_states):
            if dp[state_old] == 0:
                continue
            choices = []
            if S[i] == '?':
                choices = [0, 1]
            elif S[i] == 'A':
                choices = [0]
            else:
                choices = [1]
                
            for c in choices:
                if i >= K - 1:
                    entire = (state_old << 1) | c
                    if pal[entire]:
                        continue
                state_new = ((state_old << 1) | c) & (total_states - 1)
                new_dp[state_new] = (new_dp[state_new] + dp[state_old]) % MOD
        dp = new_dp

    ans = sum(dp) % MOD
    print(ans)

if __name__ == "__main__":
    main()