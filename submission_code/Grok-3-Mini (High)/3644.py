class Solution:
    def minimumSumSubarray(self, nums: List[int], l: int, r: int) -> int:
        # Compute prefix sums
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        min_sum = float('inf')
        
        # Iterate over all possible subarrays with length between l and r
        for i in range(len(nums)):
            start_j = i + l - 1
            end_j = min(i + r - 1, len(nums) - 1)
            if start_j <= end_j:
                for j in range(start_j, end_j + 1):
                    current_sum = prefix[j + 1] - prefix[i]
                    if current_sum > 0:
                        if current_sum < min_sum:
                            min_sum = current_sum
        
        # Check if any positive sum was found
        if min_sum == float('inf'):
            return -1
        else:
            return min_sum