class Solution:
    def largestPalindrome(self, n: int, k: int) -> str:
        prefix_length = (n + 1) // 2
        current_prefix = '9' * prefix_length
        
        while True:
            # Generate the palindrome string from current_prefix
            if n % 2 == 0:
                pal = current_prefix + current_prefix[::-1]
            else:
                pal = current_prefix + current_prefix[:-1][::-1]
            
            # Check if divisible by k
            if self.mod_k(pal, k) == 0:
                return pal
            
            # Generate next prefix
            next_prefix = self.decrement_prefix(current_prefix)
            if next_prefix is None:
                break
            current_prefix = next_prefix
        
        # According to problem constraints, there is always a solution
        return ""
    
    def decrement_prefix(self, s: str) -> str:
        s_list = list(s)
        i = len(s_list) - 1
        while i >= 0 and s_list[i] == '0':
            s_list[i] = '9'
            i -= 1
        if i == -1:
            return None
        s_list[i] = str(int(s_list[i]) - 1)
        if s_list[0] == '0':
            return None
        return ''.join(s_list)
    
    def mod_k(self, s: str, k: int) -> int:
        res = 0
        for c in s:
            res = (res * 10 + int(c)) % k
        return res