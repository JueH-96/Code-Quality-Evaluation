class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if k == 1:
            return '9' * n
        
        m = (n + 1) // 2
        max_s = '9' * m
        min_s = '1' + '0' * (m - 1)
        
        # Precompute pow(10, d) mod k for d from 0 to m
        max_pow = m
        pow10_mod = [1] * (max_pow + 1)  # pow10_mod[0] = 1
        for d in range(1, max_pow + 1):
            pow10_mod[d] = (pow10_mod[d-1] * 10) % k
        
        current_s = max_s
        while True:
            # Generate reversed part
            if n % 2 == 0:
                rev_part = current_s[::-1]
            else:
                rev_part = current_s[:-1][::-1]
            
            # Compute s_mod
            s_mod = 0
            for c in current_s:
                s_mod = (s_mod * 10 + int(c)) % k
            
            # Compute reversed_part mod
            rev_mod = 0
            for c in rev_part:
                rev_mod = (rev_mod * 10 + int(c)) % k
            
            len_rev = len(rev_part)
            pow10 = pow10_mod[len_rev]
            total_mod = (s_mod * pow10 + rev_mod) % k
            
            if total_mod == 0:
                return current_s + rev_part
            
            # Decrement current_s
            current_s = self.decrement(current_s)
            if current_s is None or current_s < min_s:
                break
        
        return ""
    
    def decrement(self, s: str) -> str:
        s_list = list(s)
        i = len(s_list) - 1
        while i >= 0:
            if s_list[i] > '0':
                s_list[i] = chr(ord(s_list[i]) - 1)
                break
            else:
                s_list[i] = '9'
                i -= 1
        
        if i < 0:
            return None
        
        # Fill the rest with 9's
        for j in range(i + 1, len(s_list)):
            s_list[j] = '9'
        
        new_s = ''.join(s_list)
        min_s = '1' + '0' * (len(s_list) - 1)
        if new_s < min_s:
            return None
        return new_s