class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        """
        We want the maximum subarray sum after optionally removing all occurrences
        of exactly one integer x (as long as that removal does not empty the array).

        Strategy:
          1) First, compute the standard maximum subarray sum (Kadane's algorithm)
             without removing anything. Call this no_remove_sum.
          2) Identify all distinct values that are negative (since removing a 
             non-negative value rarely helps, and removing a value with no negative
             impact cannot increase the max subarray sum). We will consider only
             negative distinct values as candidates to remove.
          3) For each such candidate x:
             - Perform a "skip-x Kadane": iterate through nums, completely skip
               elements equal to x, and treat the array as if those elements were
               not present at all (thus allowing us to "bridge" across them).
             - Compute the maximum subarray sum of this new "virtual" array.
             - If removing x would empty the array (i.e., all elements are x),
               we skip that case.
          4) The answer is the maximum of no_remove_sum and all these "skip-x"
             sums.

        This approach in the worst case can take O(n * k) time, where k is
        the number of distinct negative values. In practice, this is often
        acceptable for typical input distributions. 
        If the array has no negative values, the result is simply the sum of all
        elements (i.e., Kadane or the entire array).

        Constraints:
        - 1 <= len(nums) <= 10^5
        - -10^6 <= nums[i] <= 10^6
        """

        # --- 1) Standard Kadane for no removal ---
        def kadane(arr: List[int]) -> int:
            current_sum = arr[0]
            best_sum = arr[0]
            for value in arr[1:]:
                current_sum = max(value, current_sum + value)
                best_sum = max(best_sum, current_sum)
            return best_sum
        
        no_remove_sum = kadane(nums)

        # Collect distinct negative values
        negative_vals = set(val for val in nums if val < 0)
        if not negative_vals:
            # If there are no negative values at all, removing anything
            # does not help, so return the no-removal result.
            return no_remove_sum

        # --- 2) Function to do skip-x Kadane ---
        def skip_val_kadane(arr: List[int], skip_val: int) -> int:
            # We'll track whether we've used at least one element
            # (to ensure the resulting array isn't empty).
            current_sum = float('-inf')
            best_sum = float('-inf')
            used_any = False

            for num in arr:
                if num == skip_val:
                    continue
                used_any = True
                if current_sum < 0:
                    current_sum = num
                else:
                    current_sum += num
                best_sum = max(best_sum, current_sum)

            return best_sum if used_any else None

        # --- 3) Try removing each negative distinct value ---
        best_answer = no_remove_sum
        # If there's only 1 distinct value in the entire array, removing it
        # would empty the array (which is not allowed), so we skip that scenario.
        distinct_values = set(nums)
        if len(distinct_values) == 1:
            # We cannot remove the only value because that yields an empty array
            return best_answer

        # Otherwise, try removing each negative value
        for x in negative_vals:
            # If x is the only value in the array, skip it:
            # (covered by the distinct_values check above)
            # but let's just be safe if that check wasn't enough:
            if nums.count(x) == len(nums):
                continue

            candidate_sum = skip_val_kadane(nums, x)
            if candidate_sum is not None:
                best_answer = max(best_answer, candidate_sum)

        return best_answer