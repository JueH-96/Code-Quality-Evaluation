import sys

def main():
    X, Y, Z = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    n = len(S)
    
    INF = float('inf')
    prev_dp = [0, INF]
    
    for c in S:
        curr_dp = [INF, INF]
        for s_old in [0, 1]:
            if prev_dp[s_old] == INF:
                continue
            for s_prime in [0, 1]:
                # Determine key cost based on current character and s_prime
                if c == 'a':
                    key_cost = X if s_prime == 0 else Y
                else:  # 'A'
                    key_cost = Y if s_prime == 0 else X
                
                toggle_before = Z * (s_old != s_prime)
                
                # Check all possible s_new transitions
                for s_new in [0, 1]:
                    toggle_after = Z * (s_prime != s_new)
                    total_cost = toggle_before + key_cost + toggle_after
                    new_time = prev_dp[s_old] + total_cost
                    
                    if new_time < curr_dp[s_new]:
                        curr_dp[s_new] = new_time
        
        prev_dp = curr_dp
    
    print(min(prev_dp))

if __name__ == "__main__":
    main()