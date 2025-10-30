class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        if n == 1:
            for candidate in range(9, 0, -1):
                if candidate % k == 0:
                    return str(candidate)
        
        m = (n + 1) // 2
        max_exponent = m
        power10_arr = [1] * (max_exponent + 1)
        power10_arr[0] = 1 % k
        for i in range(1, max_exponent + 1):
            power10_arr[i] = (power10_arr[i-1] * 10) % k
        
        power10_rev = [0] * m
        for i in range(m):
            power10_rev[i] = power10_arr[m - 1 - i]
        
        if n % 2 == 1:
            power10_b = [0] * (m - 1)
            power10_rev_b = [0] * (m - 1)
            for i in range(m - 1):
                power10_b[i] = power10_arr[m - 2 - i]
                power10_rev_b[i] = power10_arr[i]
        else:
            power10_b = None
            power10_rev_b = None
        
        root = [9] * m
        R = 0
        revR = 0
        for i in range(m):
            R = (R + 9 * power10_rev[i]) % k
            revR = (revR + 9 * power10_arr[i]) % k
        
        if n % 2 == 1:
            residue_b = 0
            residue_rev_b = 0
            for i in range(m - 1):
                residue_b = (residue_b + 9 * power10_b[i]) % k
                residue_rev_b = (residue_rev_b + 9 * power10_rev_b[i]) % k
            last_digit = 9
        else:
            residue_b = 0
            residue_rev_b = 0
            last_digit = 0
        
        t = power10_arr[m]
        bound = 100000
        for _ in range(bound):
            if n % 2 == 0:
                total_res = (R * t + revR) % k
                if total_res == 0:
                    s = ''.join(str(d) for d in root)
                    return s + s[::-1]
            else:
                term1 = (residue_b * t) % k
                term2 = (last_digit * power10_arr[m - 1]) % k
                total_res = (term1 + term2 + residue_rev_b) % k
                if total_res == 0:
                    s = ''.join(str(d) for d in root)
                    return s + s[:-1][::-1]
            
            i = m - 1
            while i >= 0 and root[i] == 0:
                diff = 9 - 0
                R = (R + diff * power10_rev[i]) % k
                revR = (revR + diff * power10_arr[i]) % k
                if n % 2 == 1:
                    if i < m - 1:
                        residue_b = (residue_b + diff * power10_b[i]) % k
                        residue_rev_b = (residue_rev_b + diff * power10_rev_b[i]) % k
                    if i == m - 1:
                        last_digit = 9
                root[i] = 9
                i -= 1
            
            if i < 0:
                break
            
            old_val = root[i]
            new_val = old_val - 1
            diff = new_val - old_val
            R = (R + diff * power10_rev[i]) % k
            revR = (revR + diff * power10_arr[i]) % k
            if n % 2 == 1:
                if i < m - 1:
                    residue_b = (residue_b + diff * power10_b[i]) % k
                    residue_rev_b = (residue_rev_b + diff * power10_rev_b[i]) % k
                if i == m - 1:
                    last_digit = new_val
            root[i] = new_val
            
            if i == 0 and new_val == 0:
                break
        
        s = ''.join(str(d) for d in root)
        if n % 2 == 0:
            return s + s[::-1]
        else:
            return s + s[:-1][::-1]