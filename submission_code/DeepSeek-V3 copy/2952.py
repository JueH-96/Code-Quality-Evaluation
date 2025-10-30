class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        # Calculate the total sum of nums1 and nums2
        total_sum1 = sum(nums1)
        total_sum2 = sum(nums2)
        
        # If the initial sum is already <= x, return 0
        if total_sum1 <= x:
            return 0
        
        # If the total_sum2 is 0, then the sum will never decrease
        if total_sum2 == 0:
            return -1
        
        # We need to find the minimum time t such that:
        # total_sum1 + t * total_sum2 - sum_{i in S} (nums2[i] * t) <= x
        # where S is the set of indices we choose to set to 0 at some time
        
        # To minimize t, we need to maximize the sum of nums2[i] for the indices we choose to set to 0
        
        # So, we need to select a subset of indices to set to 0, and the order in which we set them to 0 matters
        
        # We can model this as a dynamic programming problem where we decide for each index whether to set it to 0 at some time
        
        # We will sort the indices based on nums2[i] in descending order to maximize the reduction
        
        # Create a list of tuples (nums2[i], nums1[i])
        pairs = list(zip(nums2, nums1))
        # Sort in descending order of nums2[i]
        pairs.sort(reverse=True, key=lambda x: x[0])
        
        # Initialize the DP table
        # dp[t][k] represents the maximum reduction achievable in t seconds using the first k indices
        # We need to find the smallest t such that total_sum1 + t * total_sum2 - dp[t][n] <= x
        
        # Initialize dp as a list of size t_max + 1, where t_max is the maximum possible time
        # Since the sum can be reduced by at most sum(nums2[i] * t) for each i, t_max can be up to x / min(nums2[i]) where nums2[i] > 0
        
        # To find t_max, we can iterate until the sum is <= x or it's clear that it's not possible
        
        # Initialize t_max as a large number, say 10^6
        t_max = 10**6
        # We will perform a binary search on t to find the minimum t
        
        left = 0
        right = t_max
        result = -1
        
        while left <= right:
            mid = (left + right) // 2
            # Calculate the total sum after mid seconds without any operations
            total = total_sum1 + mid * total_sum2
            # Now, we need to find the maximum reduction possible in mid seconds
            # We can use a greedy approach: for each second, choose the index with the largest nums2[i] that hasn't been set to 0 yet
            # So, we can sort the indices in descending order of nums2[i] and assign the first mid indices to be set to 0 at some time
            # The reduction is sum_{i=1 to mid} nums2[i] * (mid - (i-1))
            # Because the i-th index is set to 0 at the i-th second, so it contributes nums2[i] * (mid - (i-1)) to the reduction
            reduction = 0
            for i in range(min(mid, n)):
                reduction += pairs[i][0] * (mid - i)
            # Check if total - reduction <= x
            if total - reduction <= x:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result