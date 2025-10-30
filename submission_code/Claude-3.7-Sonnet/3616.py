class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        valid_count = 0
        n = len(nums)
        
        for curr in range(n):
            if nums[curr] == 0:
                # Try moving left
                if self.isValidSelection(nums.copy(), curr, -1):
                    valid_count += 1
                
                # Try moving right
                if self.isValidSelection(nums.copy(), curr, 1):
                    valid_count += 1
        
        return valid_count
    
    def isValidSelection(self, nums: List[int], curr: int, direction: int) -> bool:
        n = len(nums)
        
        while 0 <= curr < n:
            if nums[curr] == 0:
                curr += direction
            else:
                nums[curr] -= 1
                direction = -direction
                curr += direction
        
        # Check if all elements are 0
        return all(num == 0 for num in nums)