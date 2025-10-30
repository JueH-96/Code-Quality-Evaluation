import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, Q = map(int, sys.stdin.readline().split())
    operations = [tuple(map(int, sys.stdin.readline().split())) for _ in range(Q)]
    
    # Initial state: l=0, r=N+1, vl=0, vr=0
    dp = defaultdict(int)
    dp[(0, N+1, 0, 0)] = 1
    
    for idx, (P, V) in enumerate(operations):
        new_dp = defaultdict(int)
        for state, cnt in dp.items():
            l_prev, r_prev, vl_prev, vr_prev = state
            
            # Process left choice (prefix operation)
            if l_prev < r_prev:
                if P <= l_prev:
                    max_val = vl_prev
                elif P < r_prev:
                    max_val = max(vl_prev, 0)
                else:
                    max_val = max(vl_prev, 0, vr_prev)
            else:
                if P < r_prev:
                    max_val = vl_prev
                else:
                    max_val = max(vl_prev, vr_prev)
            if max_val <= V:
                new_l = max(l_prev, P)
                new_r = r_prev
                new_vl = V
                new_vr = vr_prev
                new_state = (new_l, new_r, new_vl, new_vr)
                new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
            
            # Process right choice (suffix operation)
            if l_prev < r_prev:
                if P >= r_prev:
                    max_val = vr_prev
                elif P <= l_prev:
                    max_val = max(vl_prev, 0, vr_prev)
                else:
                    max_val = max(0, vr_prev)
            else:
                if P >= r_prev:
                    max_val = vr_prev
                else:
                    max_val = max(vl_prev, vr_prev)
            if max_val <= V:
                new_l = l_prev
                new_r = min(r_prev, P)
                new_vl = vl_prev
                new_vr = V
                new_state = (new_l, new_r, new_vl, new_vr)
                new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
        
        dp = new_dp
        if not dp:
            break
    
    # Sum all possible counts
    total = sum(dp.values()) % MOD
    print(total)

if __name__ == '__main__':
    main()