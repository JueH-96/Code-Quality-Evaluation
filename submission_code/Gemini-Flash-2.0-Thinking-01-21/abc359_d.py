def is_palindrome(s):
    return s == s[::-1]

def solve():
    n, k = map(int, input().split())
    s = input()
    
    if k > n:
        q_count = s.count('?')
        print(pow(2, q_count, 998244353))
        return
        
    dp = {}
    initial_suffix = ""
    dp[initial_suffix] = 1
    
    for i in range(n):
        next_dp = {}
        for suffix in dp:
            if dp[suffix] == 0:
                continue
            possible_chars = []
            if s[i] == 'A':
                possible_chars = ['A']
            elif s[i] == 'B':
                possible_chars = ['B']
            elif s[i] == '?':
                possible_chars = ['A', 'B']
                
            for char in possible_chars:
                test_string = suffix + char
                substring_k = ""
                if len(test_string) >= k:
                    substring_k = test_string[len(test_string)-k:]
                
                is_k_palindrome = False
                if len(substring_k) == k and is_palindrome(substring_k):
                    is_k_palindrome = True
                    
                if not is_k_palindrome:
                    next_suffix = test_string[1:] if len(test_string) >= 1 else ""
                    if len(next_suffix) > k - 1:
                        next_suffix = next_suffix[len(next_suffix) - (k - 1):]
                    if next_suffix not in next_dp:
                        next_dp[next_suffix] = 0
                    next_dp[next_suffix] = (next_dp[next_suffix] + dp[suffix]) % 998244353
                    
        dp = next_dp
        
    result = 0
    for suffix in dp:
        result = (result + dp[suffix]) % 998244353
        
    print(result)

if __name__ == '__main__':
    solve()