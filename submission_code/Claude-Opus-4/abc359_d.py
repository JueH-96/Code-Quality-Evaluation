def is_palindrome(s):
    return s == s[::-1]

def solve():
    N, K = map(int, input().split())
    S = input().strip()
    
    MOD = 998244353
    
    # dp[i][mask] = number of ways to fill first i positions such that
    # the last min(i, K-1) characters match the pattern in mask
    # mask represents the last K-1 characters (0 for A, 1 for B)
    
    dp = {}
    dp[0] = {0: 1}  # Empty string
    
    for i in range(N):
        new_dp = {}
        
        for mask, count in dp.get(i, {}).items():
            # Try both 'A' and 'B' if current character is '?'
            chars_to_try = ['A', 'B'] if S[i] == '?' else [S[i]]
            
            for char in chars_to_try:
                # Build the string of last K characters
                last_chars = []
                temp_mask = mask
                for j in range(min(i, K-1)):
                    last_chars.append('A' if (temp_mask & 1) == 0 else 'B')
                    temp_mask >>= 1
                last_chars.reverse()
                last_chars.append(char)
                
                # Check if the last K characters form a palindrome
                if len(last_chars) >= K:
                    last_k = ''.join(last_chars[-K:])
                    if is_palindrome(last_k):
                        continue  # Skip this assignment
                
                # Update the mask for next state
                new_mask = 0
                for j in range(min(i+1, K-1)):
                    idx = len(last_chars) - 1 - j
                    if last_chars[idx] == 'B':
                        new_mask |= (1 << j)
                
                if i + 1 not in new_dp:
                    new_dp[i + 1] = {}
                new_dp[i + 1][new_mask] = (new_dp[i + 1].get(new_mask, 0) + count) % MOD
        
        dp[i + 1] = new_dp
    
    # Sum all valid endings
    result = 0
    for count in dp.get(N, {}).values():
        result = (result + count) % MOD
    
    print(result)

solve()