class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Initialize left_max and right_max arrays
        left_max = [0] * n
        left_max[0] = float('-inf')
        for j in range(1, n):
            left_max[j] = max(left_max[j-1], nums[j-1])
        
        right_max = [0] * n
        right_max[n-1] = float('-inf')
        for j in range(n-2, -1, -1):
            right_max[j] = max(right_max[j+1], nums[j+1])
        
        max_value = float('-inf')
        for j in range(1, n-1):
            if left_max[j] > nums[j]:
                temp = (left_max[j] - nums[j]) * right_max[j]
                if temp > max_value:
                    max_value = temp
        
        return max_value if max_value > 0 else 0