def is_palindrome(s):
    return s == s[::-1]

def count_good_strings(N, K, S):
    MOD = 998244353
    q = S.count('?')
    
    # Precompute all palindromes of length K
    palindromes = set()
    def generate_palindromes(current, length):
        if length == K:
            if is_palindrome(current):
                palindromes.add(current)
            return
        generate_palindromes(current + 'A', length + 1)
        generate_palindromes(current + 'B', length + 1)
    
    generate_palindromes('', 0)
    
    # Dynamic programming to count good strings
    dp = [0] * (1 << q)
    dp[0] = 1
    
    for i in range(N):
        if S[i] == '?':
            new_dp = [0] * (1 << q)
            for mask in range(1 << q):
                if dp[mask] > 0:
                    for bit in range(q):
                        if (mask & (1 << bit)) == 0:
                            new_mask = mask | (1 << bit)
                            new_dp[new_mask] = (new_dp[new_mask] + dp[mask]) % MOD
            dp = new_dp
        else:
            new_dp = [0] * (1 << q)
            for mask in range(1 << q):
                if dp[mask] > 0:
                    new_dp[mask] = (new_dp[mask] + dp[mask]) % MOD
            dp = new_dp
    
    # Count valid configurations
    total_good = 0
    for mask in range(1 << q):
        candidate = list(S)
        bit_index = 0
        for i in range(N):
            if candidate[i] == '?':
                if mask & (1 << bit_index):
                    candidate[i] = 'B'
                else:
                    candidate[i] = 'A'
                bit_index += 1
        candidate = ''.join(candidate)
        
        is_good = True
        for j in range(N - K + 1):
            if candidate[j:j+K] in palindromes:
                is_good = False
                break
        
        if is_good:
            total_good = (total_good + dp[mask]) % MOD
    
    return total_good

import sys
input = sys.stdin.read
def main():
    data = input().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    result = count_good_strings(N, K, S)
    print(result)

if __name__ == "__main__":
    main()