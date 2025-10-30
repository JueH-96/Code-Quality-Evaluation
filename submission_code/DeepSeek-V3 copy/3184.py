from bisect import bisect_left

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Transform the problem to find the maximum sum of a subsequence where nums[i_j] - i_j >= nums[i_{j-1}] - i_{j-1}
        # Let's define a new array a where a[i] = nums[i] - i
        a = [nums[i] - i for i in range(n)]
        
        # We need to find the maximum sum of a subsequence where a[i_j] >= a[i_{j-1}]
        # This is similar to the problem of finding the maximum sum of an increasing subsequence
        
        # We will use a list to keep track of the maximum sum for each possible a[i]
        # Since a[i] can be very large, we need to map them to a smaller range
        # We will sort the unique a[i] and assign them to indices
        
        unique_a = sorted(set(a))
        # Create a mapping from a[i] to its index in the sorted list
        a_to_index = {v: i for i, v in enumerate(unique_a)}
        
        # Initialize a list to store the maximum sum for each a[i]
        # We will use a list where dp[i] represents the maximum sum for the a[i] value
        # Since a[i] can be large, we will use a list of size len(unique_a)
        dp = [float('-inf')] * len(unique_a)
        
        # Iterate through each element in nums
        for i in range(n):
            # Get the transformed value
            current_a = a[i]
            # Find the index in the sorted unique_a
            idx = a_to_index[current_a]
            # The maximum sum up to this point is the maximum of all dp[j] where j <= idx, plus nums[i]
            # To find the maximum dp[j] where j <= idx, we can use a prefix maximum array
            # But since we are building dp on the fly, we need to find the maximum in dp[0..idx]
            # We can use a binary search to find the maximum in dp[0..idx]
            # However, to efficiently find the maximum in a range, we can use a segment tree or a Fenwick tree
            # For simplicity, we will use a list and update it as we go
            # Initialize the maximum sum to nums[i]
            max_sum = nums[i]
            # Iterate through all previous dp[j] where j <= idx
            for j in range(idx + 1):
                if dp[j] != float('-inf'):
                    max_sum = max(max_sum, dp[j] + nums[i])
            # Update dp[idx]
            dp[idx] = max(dp[idx], max_sum)
        
        # The answer is the maximum value in dp
        return max(dp)