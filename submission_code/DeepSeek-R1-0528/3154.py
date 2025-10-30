class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        left_max_arr = [0] * n
        for j in range(1, n):
            left_max_arr[j] = max(left_max_arr[j-1], nums[j-1])
        
        right_max_arr = [0] * n
        for j in range(n-2, -1, -1):
            right_max_arr[j] = max(nums[j+1], right_max_arr[j+1])
        
        ans = 0
        for j in range(n):
            candidate = (left_max_arr[j] - nums[j]) * right_max_arr[j]
            if candidate > ans:
                ans = candidate
        
        return ans