from bisect import bisect_left

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        # Transform the problem to handle the condition nums[i_j] - nums[i_{j-1}] >= i_j - i_{j-1}
        # Let's define a new array a where a[i] = nums[i] - i
        a = [nums[i] - i for i in range(n)]
        
        # We need to find the maximum sum of a subsequence where a[i_j] >= a[i_{j-1}]
        # This is similar to the problem of finding the maximum sum increasing subsequence
        
        # We will use a list to keep track of the maximum sum for each possible a[i]
        # We will use a dictionary to map a[i] to the maximum sum ending with a[i]
        # Since a[i] can be very large, we will use a sorted list to manage the keys
        
        # Initialize the sorted list with the first element
        sorted_a = []
        dp = {}
        
        for i in range(n):
            # Find the position where a[i] would be inserted to keep the list sorted
            pos = bisect_left(sorted_a, a[i])
            # The maximum sum ending with a[i] is the maximum sum of all a[j] <= a[i] plus nums[i]
            # Since we are using a sorted list, we can find the maximum sum in O(log n) time
            # We need to find the maximum dp value for all a[j] <= a[i]
            # To do this efficiently, we can maintain a list of (a[j], dp[j]) and use binary search
            # However, to keep it simple, we will iterate through the sorted_a up to pos and find the maximum dp value
            max_sum = 0
            for j in range(pos):
                if sorted_a[j] in dp:
                    max_sum = max(max_sum, dp[sorted_a[j]])
            # Add nums[i] to the max_sum
            current_sum = max_sum + nums[i]
            # Update the dp dictionary
            if a[i] in dp:
                dp[a[i]] = max(dp[a[i]], current_sum)
            else:
                dp[a[i]] = current_sum
            # Insert a[i] into the sorted_a list
            if pos < len(sorted_a):
                if sorted_a[pos] != a[i]:
                    sorted_a.insert(pos, a[i])
            else:
                sorted_a.append(a[i])
        
        # The answer is the maximum value in the dp dictionary
        return max(dp.values())