class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        # Check if the array is already non-decreasing
        is_non_decreasing = True
        for i in range(1, n):
            if nums[i] < nums[i-1]:
                is_non_decreasing = False
                break
        if is_non_decreasing:
            return n
        # If not, find the maximum possible length
        # We can try to merge subarrays to make the array non-decreasing
        # The maximum possible length is the number of elements that can be kept in non-decreasing order
        # We can use a greedy approach to find the maximum length
        # Initialize the result as 1, since at least one element can be kept
        result = 1
        # Iterate through the array and try to keep as many elements as possible in non-decreasing order
        # When we encounter a decreasing pair, we can merge them into one element
        # So, the length is reduced by 1 for each such pair
        # We can count the number of such pairs and subtract from the total length
        # But this is not sufficient, as merging can affect the subsequent elements
        # Instead, we can use a dynamic programming approach
        # dp[i] represents the maximum length of a non-decreasing array that can be made from the first i elements
        # Initialize dp[0] = 1
        # For each i, we can either keep nums[i] as is if it is greater than or equal to the last element in the current sequence
        # Or we can merge it with the previous element
        # We need to choose the option that maximizes the length
        # Initialize dp array
        dp = [1] * n
        for i in range(1, n):
            if nums[i] >= nums[i-1]:
                dp[i] = dp[i-1] + 1
            else:
                # We can merge nums[i] with nums[i-1]
                # The new element is nums[i] + nums[i-1]
                # We need to check if this new element is greater than or equal to the previous element in the sequence
                # So, we need to look back to the last element before the merge
                # To simplify, we can consider that merging reduces the length by 1
                dp[i] = dp[i-1]
        # The maximum length is the maximum value in dp array
        return max(dp)