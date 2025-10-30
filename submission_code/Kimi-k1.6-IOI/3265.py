class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        
        min_prefix = {}
        max_sum = -float('inf')
        
        for j in range(len(nums)):
            current_num = nums[j]
            # Check for current_num - k
            target = current_num - k
            if target in min_prefix:
                current_sum = prefix[j+1] - min_prefix[target]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Check for current_num + k
            target = current_num + k
            if target in min_prefix:
                current_sum = prefix[j+1] - min_prefix[target]
                if current_sum > max_sum:
                    max_sum = current_sum
            # Update the min_prefix with current_num's prefix sum (prefix[j])
            current_prefix = prefix[j]
            if current_num in min_prefix:
                if current_prefix < min_prefix[current_num]:
                    min_prefix[current_num] = current_prefix
            else:
                min_prefix[current_num] = current_prefix
        
        return max_sum if max_sum != -float('inf') else 0