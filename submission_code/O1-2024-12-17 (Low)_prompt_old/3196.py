class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        """
        We want the largest subarray size L such that we can make all elements in
        some consecutive subarray of length L identical with a total cost of at most k,
        where cost = sum of |nums[i] - T| over that subarray for the chosen T.
        
        Since the cost to make all elements in a subarray identical is minimized by
        choosing the median of that subarray as the target T (for absolute differences),
        we can sort the array and then look for a consecutive segment (in the sorted order)
        whose cost to "collapse" to its median is <= k.

        Algorithm (Binary Search on the answer):
        
        1) Sort the array.
        2) Build a prefix-sum array prefixSum, where prefixSum[i] is the sum of nums[:i].
        3) Define a helper function cost(i, L) that returns the cost of making the subarray
           nums[i : i+L] all equal to its median. If mid = i + (L - 1)//2, then
               - left side: nums[i..mid-1]
               - mid element: nums[mid]
               - right side: nums[mid+1..i+L-1]
           cost = sum(nums[mid] - nums[x] for x in left side)
                + sum(nums[x] - nums[mid] for x in right side).
           We use prefix sums to compute these in O(1).
        4) We binary-search for the maximum L in [1..n] for which there exists some subarray
           of length L with cost <= k. We slide over all subarrays of length L and check
           cost(i, L). If any subarray fits cost <= k, L is feasible; otherwise not.
        5) Return the largest feasible L.

        Complexity: O(n log n) to sort + O(n log n) for the binary search over L and the 
        sliding check, which is acceptable for n up to 10^5.
        """
        import bisect
        
        nums.sort()
        n = len(nums)
        
        # prefixSum[i] = sum of nums[:i], i.e. sum up to but not including index i
        prefixSum = [0] * (n + 1)
        for i in range(n):
            prefixSum[i + 1] = prefixSum[i] + nums[i]
        
        # Helper to compute cost to make subarray nums[i..i+L-1] identical to its median
        def cost(i: int, L: int) -> int:
            # median index in the sorted subarray
            mid = i + (L - 1) // 2
            
            # sum of left side i..mid-1
            left_sum = prefixSum[mid] - prefixSum[i]       # sum(nums[i..mid-1])
            left_size = mid - i                           # number of elements on left
            # cost for left side to raise them to nums[mid]
            left_cost = nums[mid] * left_size - left_sum
            
            # sum of right side mid+1..(i+L-1)
            right_sum = prefixSum[i + L] - prefixSum[mid + 1]  # sum(nums[mid+1..i+L-1])
            right_size = (i + L - 1) - mid
            # cost for right side to lower to nums[mid]
            right_cost = right_sum - nums[mid] * right_size
            
            return left_cost + right_cost
        
        # Check feasibility of length L: can we find a subarray of length L with cost <= k?
        def can_do(L: int) -> bool:
            if L == 1:
                return True  # No cost needed for a single element
            for start in range(0, n - L + 1):
                if cost(start, L) <= k:
                    return True
            return False
        
        # Binary search for the maximum L in [1..n]
        left, right = 1, n
        ans = 1
        while left <= right:
            mid = (left + right) // 2
            if can_do(mid):
                ans = mid  # mid is feasible
                left = mid + 1
            else:
                right = mid - 1
        
        return ans