class Solution:
    def minEnd(self, n: int, x: int) -> int:
        nums = [x]
        for i in range(1, n):
            curr = nums[-1] + 1
            temp_and = nums[0]
            for j in range(1, len(nums)):
                temp_and &= nums[j]
            
            while (temp_and & curr) != x:
                curr += 1
            nums.append(curr)
        
        return nums[-1]