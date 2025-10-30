class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        total = sum(nums)
        # If the total sum is odd, no partition can have even difference
        if total % 2 != 0:
            return 0
        
        count = 0
        current_sum = 0
        # loop until the second to last element because both subarrays must be non-empty
        for i in range(len(nums) - 1):
            current_sum += nums[i]
            # left_sum is current_sum, right_sum = total - current_sum
            # The difference (left_sum - right_sum) is even if both sums have the same parity.
            if current_sum % 2 == (total - current_sum) % 2:
                count += 1
        
        return count