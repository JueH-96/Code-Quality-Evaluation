class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        
        def simulate(start: int, direction: int) -> bool:
            curr = start
            nums_copy = nums.copy()
            
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += direction
                else:
                    nums_copy[curr] -= 1
                    direction *= -1
                    curr += direction
            
            return all(num == 0 for num in nums_copy)
        
        count = 0
        for i in range(n):
            if nums[i] == 0:
                if simulate(i, 1):  # Try moving right
                    count += 1
                if simulate(i, -1):  # Try moving left
                    count += 1
        
        return count