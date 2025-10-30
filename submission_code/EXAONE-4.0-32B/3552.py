class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            max_digit = 9
            for d in range(max_digit, 0, -1):
                if d % k == 0:
                    return str(d)
        
        h = (n + 1) // 2
        c = [0] * h
        
        if n % 2 == 0:
            for i in range(h):
                exp1 = 2 * h - 1 - i
                exp2 = i
                term1 = pow(10, exp1, k)
                term2 = pow(10, exp2, k)
                c[i] = (term1 + term2) % k
        else:
            for i in range(h):
                if i < h - 1:
                    exp1 = n - 1 - i
                    exp2 = i
                    term1 = pow(10, exp1, k)
                    term2 = pow(10, exp2, k)
                    c[i] = (term1 + term2) % k
                else:
                    exp1 = n - 1 - i
                    c[i] = pow(10, exp1, k) % k
        
        dp = [False] * k
        dp[0] = True
        back = [None] * (h + 1)
        
        for i in range(h):
            new_dp = [False] * k
            back_i_plus1 = [None] * k
            for r_old in range(k):
                if not dp[r_old]:
                    continue
                for d in range(9, -1, -1):
                    if i == 0 and d == 0:
                        continue
                    add_val = (d * c[i]) % k
                    r_new = (r_old + add_val) % k
                    if not new_dp[r_new]:
                        new_dp[r_new] = True
                        back_i_plus1[r_new] = (d, r_old)
            dp = new_dp
            back[i + 1] = back_i_plus1
        
        if not dp[0]:
            return ""
        
        r = 0
        digits = []
        for i in range(h - 1, -1, -1):
            d, prev_r = back[i + 1][r]
            digits.append(str(d))
            r = prev_r
        left_half = ''.join(digits[::-1])
        
        if n % 2 == 0:
            res_str = left_half + left_half[::-1]
        else:
            res_str = left_half + left_half[:-1][::-1]
        
        return res_str