class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        max_sum = 0
        found = False
        
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i+k]
            if len(set(subarray)) >= m:
                found = True
                current_sum = sum(subarray)
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum if found else 0