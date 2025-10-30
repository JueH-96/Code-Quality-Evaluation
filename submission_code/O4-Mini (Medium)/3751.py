from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        # Count how many k's are already in the array
        total_k = 0
        for x in nums:
            if x == k:
                total_k += 1
        
        # best additional gain from applying one operation
        best_gain = 0
        
        # Consider targeting each value v in nums (v != k)
        # We'll choose x = k - v, so v + x == k.
        # In a chosen subarray, elements == v give +1 gain, elements == k give -1 (we lose them).
        distinct_vals = set(nums)
        if k in distinct_vals:
            distinct_vals.remove(k)
        
        # For each candidate v, run Kadane's to find the maximum subarray sum
        for v in distinct_vals:
            current = 0
            best_for_v = 0
            # Build w_t on the fly: +1 if nums[t]==v, -1 if nums[t]==k, else 0
            for a in nums:
                delta = 1 if a == v else (-1 if a == k else 0)
                # Kadane step
                current = max(delta, current + delta)
                best_for_v = max(best_for_v, current)
            best_gain = max(best_gain, best_for_v)
        
        # The answer is original count of k plus the best gain (or zero if no positive gain)
        return total_k + best_gain