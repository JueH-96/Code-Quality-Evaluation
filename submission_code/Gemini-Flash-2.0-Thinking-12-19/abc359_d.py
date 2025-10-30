def is_palindrome(s):
    return s == s[::-1]

def solve():
    n, k = map(int, input().split())
    s = input()
    
    dp = {}
    dp[""] = 1
    
    for i in range(n):
        next_dp = {}
        for suffix in dp:
            count = dp[suffix]
            possible_chars = []
            if s[i] == 'A':
                possible_chars = ['A']
            elif s[i] == 'B':
                possible_chars = ['B']
            elif s[i] == '?':
                possible_chars = ['A', 'B']
                
            for char in possible_chars:
                current_substring = suffix + char
                is_k_palindrome = False
                if len(current_substring) >= k:
                    substring_to_check = current_substring[len(current_substring)-k:]
                    if is_palindrome(substring_to_check):
                        is_k_palindrome = True
                        
                if not is_k_palindrome:
                    next_suffix = current_substring[max(0, len(current_substring) - (k - 1)):]
                    next_dp[next_suffix] = next_dp.get(next_suffix, 0) + count
                    next_dp[next_suffix] %= 998244353
                    
        dp = next_dp
        
    result = 0
    for suffix in dp:
        result += dp[suffix]
        result %= 998244353
        
    print(result)

if __name__ == '__main__':
    solve()