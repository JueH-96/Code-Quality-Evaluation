class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        m = (n + 1) // 2
        minimal_s = '1' + '0' * (m - 1)
        current_s = '9' * m
        
        def subtract_one(s):
            s_list = list(s)
            i = len(s_list) - 1
            while i >= 0:
                if s_list[i] == '0':
                    s_list[i] = '9'
                    i -= 1
                else:
                    s_list[i] = str(int(s_list[i]) - 1)
                    if i == 0 and s_list[i] == '0':
                        return None
                    break
            new_s = ''.join(s_list)
            if new_s[0] == '0' or len(new_s) < len(s):
                return None
            return new_s
        
        while current_s is not None and current_s >= minimal_s:
            # Compute s_mod
            s_mod = 0
            for c in current_s:
                s_mod = (s_mod * 10 + int(c)) % k
            
            # Compute reversed_part_str
            if n % 2 == 0:
                reversed_part_str = current_s[::-1]
            else:
                reversed_part_str = current_s[:-1][::-1]
            
            # Compute rev_mod
            rev_mod = 0
            for c in reversed_part_str:
                rev_mod = (rev_mod * 10 + int(c)) % k
            
            # Compute pow_10_p
            if n % 2 == 0:
                p = m
            else:
                p = m - 1
            pow_10_p = pow(10, p, k)
            
            total_mod = (s_mod * pow_10_p + rev_mod) % k
            if total_mod == 0:
                # Construct the palindrome
                if n % 2 == 0:
                    palindrome = current_s + reversed_part_str
                else:
                    palindrome = current_s + reversed_part_str
                return palindrome
            
            # Subtract one
            new_s = subtract_one(current_s)
            if new_s is None:
                current_s = None
            else:
                current_s = new_s
        
        # If no solution found (shouldn't happen as per problem statement)
        return ''