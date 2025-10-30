class Solution:
    def countValidSelections(self, nums: List[int]) -> int:
        n = len(nums)
        
        def simulate(start: int, direction: int, nums_copy: List[int]) -> bool:
            curr = start
            
            while 0 <= curr < n:
                if nums_copy[curr] == 0:
                    curr += direction
                else:
                    nums_copy[curr] -= 1
                    direction = -direction
                    curr += direction
            
            # Check if all elements are 0
            return all(x == 0 for x in nums_copy)
        
        count = 0
        # Try all possible starting positions where nums[curr] == 0
        for start in range(n):
            if nums[start] == 0:
                # Try both directions: -1 (left) and 1 (right)
                for direction in [-1, 1]:
                    # Create a copy of nums for simulation
                    nums_copy = nums.copy()
                    if simulate(start, direction, nums_copy):
                        count += 1
        
        return count