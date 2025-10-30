import sys
from collections import defaultdict

def main():
    MOD = 998244353
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    Q = int(input[idx])
    idx += 1
    operations = []
    for _ in range(Q):
        P = int(input[idx])
        idx += 1
        V = int(input[idx])
        idx += 1
        operations.append((P, V))
    
    current_dp = defaultdict(int)
    initial_state = (0, 0, N + 1, 0)
    current_dp[initial_state] = 1
    
    for P, V in operations:
        next_dp = defaultdict(int)
        for (a, b, R, L), cnt in current_dp.items():
            # Check type 1 (prefix)
            if R <= P:
                req = max(a, b)
            else:
                req = a
            if V >= req:
                new_a = max(a, V)
                new_b = b
                new_R = R
                new_L = max(L, P)
                next_dp[(new_a, new_b, new_R, new_L)] = (next_dp[(new_a, new_b, new_R, new_L)] + cnt) % MOD
            
            # Check type 2 (suffix)
            if L >= P:
                req = max(a, b)
            else:
                req = b
            if V >= req:
                new_a = a
                new_b = max(b, V)
                new_R = min(R, P)
                new_L = L
                next_dp[(new_a, new_b, new_R, new_L)] = (next_dp[(new_a, new_b, new_R, new_L)] + cnt) % MOD
        
        current_dp = next_dp
        if not current_dp:
            break  # No valid sequences, exit early
    
    print(sum(current_dp.values()) % MOD)

if __name__ == "__main__":
    main()