import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    Q = int(input[idx]); idx += 1
    operations = []
    for _ in range(Q):
        P = int(input[idx]); idx += 1
        V = int(input[idx]); idx += 1
        operations.append((P, V))
    
    prev_dp = defaultdict(int)
    # Initial state: L=0, R=N+1, valL=0, valR=0
    prev_dp[(0, N+1, 0, 0)] = 1
    
    for P, V in operations:
        curr_dp = defaultdict(int)
        for (L, R, valL, valR), cnt in prev_dp.items():
            # Check prefix choice
            if L >= P:
                max_val = valL
            else:
                if R <= P:
                    max_val = max(valL, valR)
                else:
                    max_val = valL
            if max_val <= V:
                new_L = max(L, P)
                new_valL = V
                new_R = R
                new_valR = valR
                curr_dp[(new_L, new_R, new_valL, new_valR)] = (curr_dp[(new_L, new_R, new_valL, new_valR)] + cnt) % MOD
            
            # Check suffix choice
            max_val = 0
            if P <= L:
                max_val = valL
            if R <= P:
                max_val = max(max_val, valR)
            if max_val <= V:
                new_R = min(R, P)
                new_valR = V
                new_L = L
                new_valL = valL
                curr_dp[(new_L, new_R, new_valL, new_valR)] = (curr_dp[(new_L, new_R, new_valL, new_valR)] + cnt) % MOD
        prev_dp = curr_dp
        
        # If no possible states, break early
        if not prev_dp:
            print(0)
            return
    
    print(sum(prev_dp.values()) % MOD)

if __name__ == '__main__':
    main()