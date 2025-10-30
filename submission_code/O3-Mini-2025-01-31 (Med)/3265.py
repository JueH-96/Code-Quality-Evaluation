from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Build prefix sum array: prefix[i] is sum(nums[0:i])
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        # Dictionary to maintain the minimum prefix sum for an index where the subarray could start,
        # keyed by the actual number at that start index.
        best_prefix = {}
        
        # Initialize with the very first element (i = 0).
        # For a subarray starting at index 0, we subtract prefix[0] (which is 0).
        best_prefix[nums[0]] = prefix[0]
        
        res = None
        
        # Iterate over potential end indices j starting from 1.
        for j in range(1, n):
            candidate_sum = None
            
            # Condition: |nums[i] - nums[j]| == k.
            # This gives two cases:
            # Case 1: nums[i] = nums[j] + k,
            # Case 2: nums[i] = nums[j] - k.
            
            # Case 1:
            target1 = nums[j] + k
            if target1 in best_prefix:
                candidate_sum = prefix[j+1] - best_prefix[target1]
            
            # Case 2:
            target2 = nums[j] - k
            if target2 in best_prefix:
                cand2 = prefix[j+1] - best_prefix[target2]
                if candidate_sum is None or cand2 > candidate_sum:
                    candidate_sum = cand2
            
            # Update the result if a valid candidate was found.
            if candidate_sum is not None:
                if res is None or candidate_sum > res:
                    res = candidate_sum
            
            # Now, add the current index j as a potential starting point for future subarrays.
            # We want to keep the minimum prefix[j] for a given nums[j] as that maximizes the subarray sum.
            if nums[j] in best_prefix:
                best_prefix[nums[j]] = min(best_prefix[nums[j]], prefix[j])
            else:
                best_prefix[nums[j]] = prefix[j]
        
        return res if res is not None else 0

# Sample test cases to verify the solution:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumSubarraySum([1, 2, 3, 4, 5, 6], 1))  # Expected output: 11
    print(sol.maximumSubarraySum([-1, 3, 2, 4, 5], 3))    # Expected output: 11
    print(sol.maximumSubarraySum([-1, -2, -3, -4], 2))     # Expected output: -6