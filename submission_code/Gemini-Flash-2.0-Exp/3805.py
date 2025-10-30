class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        n = len(s)
        max_active = s.count('1')

        for i in range(n):
            for j in range(i, n):
                temp_s = list(s)
                
                # Trade 1: Convert 1s to 0s
                start_ones = -1
                end_ones = -1
                
                for k in range(n - 1):
                    if temp_s[k] == '0' and temp_s[k+1] == '1':
                        start_ones_temp = k + 1
                        
                        end_ones_temp = -1
                        for l in range(start_ones_temp, n):
                            if l == n - 1 and temp_s[l] == '1':
                                end_ones_temp = l
                                break
                            elif temp_s[l] == '1' and l + 1 < n and temp_s[l+1] == '0':
                                end_ones_temp = l
                                break
                        
                        if end_ones_temp != -1:
                            start_ones = start_ones_temp
                            end_ones = end_ones_temp
                            break
                
                if start_ones != -1 and end_ones != -1:
                    temp_s_copy = list(s)
                    
                    for k in range(start_ones, end_ones + 1):
                        temp_s_copy[k] = '0'
                    
                    # Trade 2: Convert 0s to 1s
                    start_zeros = -1
                    end_zeros = -1
                    
                    for k in range(n - 1):
                        if temp_s_copy[k] == '1' and temp_s_copy[k+1] == '0':
                            start_zeros_temp = k + 1
                            
                            end_zeros_temp = -1
                            for l in range(start_zeros_temp, n):
                                if l == n - 1 and temp_s_copy[l] == '0':
                                    end_zeros_temp = l
                                    break
                                elif temp_s_copy[l] == '0' and l + 1 < n and temp_s_copy[l+1] == '1':
                                    end_zeros_temp = l
                                    break
                            
                            if end_zeros_temp != -1:
                                start_zeros = start_zeros_temp
                                end_zeros = end_zeros_temp
                                break
                    
                    if start_zeros != -1 and end_zeros != -1:
                        for k in range(start_zeros, end_zeros + 1):
                            temp_s_copy[k] = '1'
                        
                        max_active = max(max_active, temp_s_copy.count('1'))
        
        return max_active