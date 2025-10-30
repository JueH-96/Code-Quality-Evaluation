def solve():
    MOD = 998244353
    S = input()
    N = len(S)
    
    # dp[pos][has_first][has_second][has_third] = count of valid strings
    dp = {}
    
    def rec(pos, has_first, has_second, has_third):
        if (pos, has_first, has_second, has_third) in dp:
            return dp[(pos, has_first, has_second, has_third)]
            
        if pos == N:
            return 1 if not (has_first and has_second and has_third) else 0
            
        ans = 0
        curr = S[pos]
        
        if curr == '?':
            # Try uppercase
            for c in range(26):
                ch = chr(ord('A') + c)
                
                # If we don't have first yet, this can be first
                new_first = has_first
                new_second = has_second
                new_third = has_third
                
                if not has_first:
                    new_first = ch
                elif has_first == ch and not has_second:
                    new_second = True
                elif has_first == ch and has_second and has_third:
                    new_third = True
                    
                ans = (ans + rec(pos + 1, new_first, new_second, new_third)) % MOD
                
            # Try lowercase
            for c in range(26):
                ch = chr(ord('a') + c)
                
                new_first = has_first
                new_second = has_second
                new_third = has_third
                
                if has_first and has_second and not has_third:
                    new_third = True
                    
                ans = (ans + rec(pos + 1, new_first, new_second, new_third)) % MOD
                
        else:
            # Fixed character
            new_first = has_first
            new_second = has_second
            new_third = has_third
            
            if curr.isupper():
                if not has_first:
                    new_first = curr
                elif has_first == curr and not has_second:
                    new_second = True
                elif has_first == curr and has_second and has_third:
                    new_third = True
            else:
                if has_first and has_second and not has_third:
                    new_third = True
                    
            ans = rec(pos + 1, new_first, new_second, new_third)
            
        dp[(pos, has_first, has_second, has_third)] = ans
        return ans
        
    print(rec(0, False, False, False))

solve()