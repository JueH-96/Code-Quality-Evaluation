class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        s = set(nums)
        if k not in s:
            return -1

        while max(nums) > k:
            
            valid_h = -1
            
            for h in sorted(list(set(nums)),reverse=True):
                
                greater_than_h = [x for x in nums if x > h]
                if not greater_than_h:
                    continue
                
                if len(set(greater_than_h)) ==1:
                    valid_h = h
                    break
            
            if valid_h == -1:
                return -1
            
            for i in range(len(nums)):
                if nums[i] > valid_h:
                    nums[i] = valid_h
            count += 1
        return count