class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        full_copies = target // total_sum
        target %= total_sum
        
        nums.extend(nums)
        n = len(nums)
        min_length = float('inf')
        current_sum = 0
        start = 0
        
        for end in range(n):
            current_sum += nums[end]
            while current_sum >= target:
                if current_sum == target:
                    min_length = min(min_length, end - start + 1)
                current_sum -= nums[start]
                start += 1
        
        return full_copies * n + min_length if min_length != float('inf') else -1