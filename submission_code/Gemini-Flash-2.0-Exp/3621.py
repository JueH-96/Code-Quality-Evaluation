class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        ops = 0
        s = set(nums)
        
        if k not in nums and len(s) > 0:
            return -1
        
        while any(x != k for x in nums):
            valid_h = []
            for h in range(1, 101):
                greater_than_h = [x for x in nums if x > h]
                if not greater_than_h:
                    continue
                
                if all(x == greater_than_h[0] for x in greater_than_h):
                    valid_h.append(h)
            
            if not valid_h:
                return -1
            
            h = max(valid_h)
            
            for i in range(len(nums)):
                if nums[i] > h:
                    nums[i] = h
            
            ops += 1
        
        return ops