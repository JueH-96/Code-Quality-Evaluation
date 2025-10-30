class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        s = set()
        
        for num in nums:
            if num > k:
                s.add(num)
        
        if len(s) > 1:
            return -1
            
        if len(s) == 0:
            for num in nums:
                if num != k:
                    return -1
            return 0
            
        max_val = s.pop()
        
        
        while max_val > k:
            count += 1
            max_val = 0
            for num in nums:
                if num > k and num < max_val:
                    max_val = num
                elif num > k and max_val == 0:
                    max_val = num
            if max_val == 0:
                for num in nums:
                    if num != k:
                        return -1
                return count
            
            
            
        
        for num in nums:
            if num != k:
                return -1
        return count