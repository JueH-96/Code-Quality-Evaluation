from collections import defaultdict

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if not nums or numOperations == 0:
            return 0
        
        # Sort the array to facilitate the sliding window approach
        nums.sort()
        n = len(nums)
        max_freq = 1
        left = 0
        total = 0
        
        for right in range(n):
            # The target is to make all elements in the window equal to nums[right]
            # The total number of operations needed is the sum of (nums[right] - nums[i]) for all i in [left, right]
            # Since we can only perform numOperations operations, we need to ensure that the total operations do not exceed numOperations * k
            # However, since each operation can add any value in [-k, k], the total change per operation is up to 2k (from -k to +k)
            # But to maximize the frequency, we should try to make as many elements as possible equal to nums[right]
            # So, we calculate the total number of operations needed to make all elements in the window equal to nums[right]
            # The total operations needed is (nums[right] - nums[left]) * (right - left + 1) - sum(nums[left:right+1])
            # Wait, no. The total operations needed is the sum of (nums[right] - nums[i]) for i in [left, right]
            # Which is nums[right] * (right - left + 1) - sum(nums[left:right+1])
            # But since we can only perform numOperations operations, and each operation can change any element by up to 2k (from -k to +k)
            # So, the total possible change is numOperations * 2k
            # So, we need to find the largest window where the sum of (nums[right] - nums[i]) for i in [left, right] <= numOperations * 2k
            # So, we can use a sliding window approach to find the maximum window size where the sum of (nums[right] - nums[i]) for i in [left, right] <= numOperations * 2k
            # So, we can calculate the sum of (nums[right] - nums[i]) for i in [left, right] as nums[right] * (right - left + 1) - sum(nums[left:right+1])
            # So, we need to maintain the sum of the window
            # So, we can use a variable to keep track of the sum of the window
            # So, we can calculate the required operations as nums[right] * (right - left + 1) - total
            # If this is greater than numOperations * 2k, we need to move the left pointer
            # Otherwise, we can update the max_freq
            # So, we can proceed as follows:
            total += nums[right]
            while (nums[right] * (right - left + 1) - total) > numOperations * 2 * k:
                total -= nums[left]
                left += 1
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq