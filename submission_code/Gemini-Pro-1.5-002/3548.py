class Solution:
    def countGoodIntegers(self, n: int, k: int) -> int:
        MOD = 10**9 + 7

        def is_palindrome(num):
            s = str(num)
            return s == s[::-1]

        def count_good_integers(n, k):
            count = 0
            for i in range(10**(n - 1), 10**n):
                if self.is_good_integer(str(i), k):
                    count += 1
            return count
        
        def is_good_integer(self, num_str, k):
            from itertools import permutations
            for p in permutations(num_str):
                num = int("".join(p))
                if num % k == 0 and is_palindrome(num):
                    return True
            return False

        
        if n == 1:
            ans = 0
            for i in range(1, 10):
                if i % k == 0:
                    ans += 1
            return ans
        
        
        dp = {}

        def solve(idx, count, rem, tight):
            if idx == n:
                if rem == 0:
                    
                    return 1
                return 0
            
            if (idx, count, rem, tight) in dp:
                return dp[(idx, count, rem, tight)]
            
            ans = 0
            limit = 9 if tight == 0 else int(num_str[idx])
            
            for digit in range(limit + 1):
                
                new_tight = 0
                if tight == 1 and digit == limit:
                    new_tight = 1
                    
                if idx == 0 and digit == 0:
                    continue
                
                
                ans += solve(idx + 1, count + Counter(str(digit)), (rem * 10 + digit) % k, new_tight)
                ans %= MOD
                
            dp[(idx, count, rem, tight)] = ans
            return ans
        
        from collections import Counter
        ans = 0
        
        for num_str in [str(i) for i in range(10**(n-1), 10**n)]:
            
            
            counts = Counter(num_str)
            
            
            def check(counts, k):
                
                
                def generate_permutations(counts):
                    
                    res = []
                    
                    def backtrack(current_perm):
                        if len(current_perm) == sum(counts.values()):
                            res.append(int("".join(current_perm)))
                            return
                        
                        for digit in counts:
                            if counts[digit] > 0:
                                counts[digit] -= 1
                                backtrack(current_perm + [digit])
                                counts[digit] += 1
                    
                    backtrack([])
                    return res
                
                perms = generate_permutations(counts)
                
                for perm in perms:
                    if perm % k == 0 and is_palindrome(perm):
                        return True
                return False

            if check(counts, k):
                ans += 1
        return ans