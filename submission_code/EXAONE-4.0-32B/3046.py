class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        patterns = [('0', '0'), ('2', '5'), ('5', '0'), ('7', '5')]
        max_kept = 0
        
        for d1, d2 in patterns:
            j = -1
            for idx in range(n-1, -1, -1):
                if num[idx] == d2:
                    j = idx
                    break
            if j == -1:
                continue
                
            i_index = -1
            for idx in range(j-1, -1, -1):
                if num[idx] == d1:
                    i_index = idx
                    break
                    
            if i_index == -1:
                continue
                
            kept = i_index + 2
            if kept > max_kept:
                max_kept = kept
                
        if max_kept == 0:
            if '0' in num:
                max_kept = 1
            else:
                max_kept = 0
                
        return n - max_kept