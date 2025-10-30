class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        m = (n + 1) // 2
        if m == 0:
            return "0"  # Not possible as per problem constraints
        
        weights = [0] * m
        for i in range(m):
            if n % 2 == 0:
                exp1 = i
                exp2 = (n - 1) - i
            else:
                if i == m - 1:
                    exp1 = None
                    exp2 = (n - 1) - i
                else:
                    exp1 = i
                    exp2 = (n - 1) - i
            
            pow1 = pow(10, exp1, k) if exp1 is not None else 0
            pow2 = pow(10, exp2, k) if exp2 is not None else 0
            
            if (n % 2 != 0) and (i == m - 1):
                weights[i] = pow2 % k
            else:
                weights[i] = (pow1 + pow2) % k
        
        possible = [[False] * k for _ in range(m + 1)]
        possible[m][0] = True
        
        for i in range(m - 1, -1, -1):
            for s in range(k):
                found = False
                start = 1 if i == 0 else 0
                for d in range(start, 10):
                    contrib = (d * weights[i]) % k
                    req_sum_next = (s - contrib) % k
                    if possible[i + 1][req_sum_next]:
                        found = True
                        break
                possible[i][s] = found
        
        digits = []
        current_sum = 0
        
        for i in range(m):
            found = False
            for d in range(9, -1, -1):
                if i == 0 and d == 0:
                    continue
                contrib = (d * weights[i]) % k
                new_sum = (current_sum + contrib) % k
                required_sum_remaining = (-new_sum) % k
                if possible[i + 1][required_sum_remaining]:
                    digits.append(str(d))
                    current_sum = new_sum
                    found = True
                    break
            if not found:
                return ""
        
        first_half = ''.join(digits)
        if n % 2 == 0:
            palindrome = first_half + first_half[::-1]
        else:
            palindrome = first_half + first_half[:-1][::-1]
        return palindrome