class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        if k == 1:
            return all(x == 0 for x in nums)
        
        # Create a difference array to apply the range increment technique
        diff = [0] * (n + 1)
        
        for i in range(n):
            if nums[i] + diff[i] > 0:
                decrement = nums[i] + diff[i]
                if i + k > n:
                    return False
                diff[i] -= decrement
                diff[i + k] += decrement
        
        return True