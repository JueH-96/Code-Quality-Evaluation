class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        value_to_index = {}
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]
        
        max_sum = float('-inf')
        found = False
        
        for j in range(n):
            # Check for starting positions with value nums[j] + k or nums[j] - k
            for target in [nums[j] + k, nums[j] - k]:
                if target in value_to_index:
                    i = value_to_index[target]
                    subarray_sum = prefix_sum[j + 1] - prefix_sum[i]
                    max_sum = max(max_sum, subarray_sum)
                    found = True
            
            # Update value_to_index for nums[j]
            if nums[j] not in value_to_index:
                value_to_index[nums[j]] = j
            else:
                if prefix_sum[j] < prefix_sum[value_to_index[nums[j]]]:
                    value_to_index[nums[j]] = j
        
        return max_sum if found else 0