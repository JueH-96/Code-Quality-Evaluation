from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Create prefix sum array where prefix[i+1] equals the sum of nums[0..i]
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Dictionary to map a number (nums[i]) to the smallest prefix sum (prefix[i])
        # observed so far for indices corresponding to that number. This helps us
        # maximize the subarray sum: subarray sum from i to j is prefix[j+1]-prefix[i].
        best = {}
        best[nums[0]] = prefix[0]  # For i = 0, prefix[0] is 0
        
        ans = None  # Will store the maximum good subarray sum encountered
        
        # Iterate j from 1 to n-1, treating nums[j] as the last element of the subarray.
        for j in range(1, n):
            # We need a starting index i (< j) such that either:
            #   nums[i] == nums[j] - k   OR   nums[i] == nums[j] + k,
            # which ensures |nums[i] - nums[j]| == k.
            candidate1 = nums[j] - k
            candidate2 = nums[j] + k
            if candidate1 in best:
                subarray_sum = prefix[j+1] - best[candidate1]
                if ans is None or subarray_sum > ans:
                    ans = subarray_sum
            if candidate2 in best:
                subarray_sum = prefix[j+1] - best[candidate2]
                if ans is None or subarray_sum > ans:
                    ans = subarray_sum
            
            # Now, update our dictionary with the current index j as a potential start.
            # We store the minimal prefix sum for the element nums[j].
            if nums[j] in best:
                best[nums[j]] = min(best[nums[j]], prefix[j])
            else:
                best[nums[j]] = prefix[j]
        
        # Return 0 if no good subarray was found; otherwise, return the maximum sum.
        return ans if ans is not None else 0