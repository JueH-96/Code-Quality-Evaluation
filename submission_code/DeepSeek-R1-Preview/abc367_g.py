import sys
from collections import defaultdict

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    M = int(input[1])
    K = int(input[2])
    A = list(map(int, input[3:3+N]))
    
    # Initialize DP: list of M defaultdicts, each for XOR counts
    dp = [defaultdict(int) for _ in range(M)]
    
    for a in A:
        # Temporary storage for new contributions
        temp = [defaultdict(int) for _ in range(M)]
        
        for m in range(M):
            current_dict = dp[m]
            if not current_dict:
                continue
            new_mod = (m + 1) % M
            for x, cnt in current_dict.items():
                new_x = x ^ a
                temp[new_mod][new_x] = (temp[new_mod][new_x] + cnt) % MOD
        
        # Merge temp into dp
        for m in range(M):
            temp_dict = temp[m]
            if not temp_dict:
                continue
            for x, cnt in temp_dict.items():
                dp[m][x] = (dp[m][x] + cnt) % MOD
        
        # Add the single element a to mod 1
        mod = 1 % M
        dp[mod][a] = (dp[mod][a] + 1) % MOD
    
    # Compute the total sum for mod 0
    total = 0
    for x, cnt in dp[0].items():
        # Compute x^K mod MOD
        xk = pow(x, K, MOD)
        total = (total + xk * cnt) % MOD
    
    print(total % MOD)

if __name__ == '__main__':
    main()