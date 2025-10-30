MOD = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data: 
        print(0)
        return
    n, k = map(int, data[0].split())
    s = data[1].strip()
    
    if s[0] == '?':
        dp = {'A': 1, 'B': 1}
    else:
        dp = {s[0]: 1}
        
    if n == 1:
        print(sum(dp.values()) % MOD)
        return
        
    for i in range(1, n):
        next_dp = {}
        choices = ['A', 'B'] if s[i] == '?' else [s[i]]
        for state, cnt in dp.items():
            for c in choices:
                if len(state) < k - 1:
                    new_state = state + c
                    next_dp[new_state] = (next_dp.get(new_state, 0) + cnt) % MOD
                else:
                    substr = state + c
                    if substr == substr[::-1]:
                        continue
                    else:
                        new_state = state[1:] + c
                        next_dp[new_state] = (next_dp.get(new_state, 0) + cnt) % MOD
        dp = next_dp
        if not dp:
            break
            
    ans = sum(dp.values()) % MOD
    print(ans)

if __name__ == "__main__":
    main()