class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        max_or = 0
        
        # Iterate through each number in the array
        for i in range(len(nums)):
            # Calculate the current number after multiplying by 2 up to k times
            current_num = nums[i]
            for j in range(k + 1):
                # Calculate the OR value with the current number
                current_or = max_or | current_num
                max_or = max(max_or, current_or)
                # Multiply the current number by 2 for the next iteration
                current_num *= 2
        
        return max_or