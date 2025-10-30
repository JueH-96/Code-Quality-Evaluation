class Solution:
    def minLength(self, s: str, numOps: int) -> int:
        n = len(s)
        min_len = n
        for char_to_make in ['0', '1']:
            for i in range(n):
                for j in range(i, n):
                    temp_s = list(s)
                    ops_needed = 0
                    for k in range(i, j + 1):
                        if temp_s[k] != char_to_make:
                            ops_needed += 1
                    if ops_needed <= numOps:
                        
                        for k in range(i, j+1):
                            if temp_s[k] != char_to_make:
                                temp_s[k] = char_to_make
                        
                        
                        max_len = 0
                        curr_len = 0
                        
                        if len(temp_s) > 0:
                            curr_char = temp_s[0]
                            curr_len = 1
                            max_len = 1
                        
                        for k in range(1, len(temp_s)):
                            if temp_s[k] == curr_char:
                                curr_len += 1
                            else:
                                max_len = max(max_len, curr_len)
                                curr_char = temp_s[k]
                                curr_len = 1
                        max_len = max(max_len, curr_len)
                        
                        min_len = min(min_len, max_len)
        
        
        
        if min_len == n:
            max_len = 0
            curr_len = 0
            if len(s) > 0:
                curr_char = s[0]
                curr_len = 1
                max_len = 1
            for k in range(1, len(s)):
                if s[k] == curr_char:
                    curr_len += 1
                else:
                    max_len = max(max_len, curr_len)
                    curr_char = s[k]
                    curr_len = 1
            max_len = max(max_len, curr_len)
            min_len = min(min_len, max_len)
        
        return min_len