from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)

        # Step 1: Calculate the initial frequency of k
        total_k = 0
        for num in nums:
            if num == k:
                total_k += 1

        # Step 2: Find distinct values in nums
        # These are the potential values 'v' that we might transform into 'k'.
        distinct_values = set(nums)

        # Initialize max_overall_kadane_sum.
        # It tracks the maximum possible value of (N_v(i, j) - N_k(i, j))
        # over all possible choices of v (from distinct_values) and all subarrays [i, j].
        # This value represents the maximum possible *change* in frequency
        # inside the modified subarray [i, j].
        # Initialize with a very small number.
        # The constraints guarantee n >= 1, so distinct_values is not empty.
        max_overall_kadane_sum = -float('inf')

        # Iterate through each distinct value v present in nums.
        # We consider the operation where elements equal to v within some subarray [i, j]
        # are transformed into k by adding x = k - v.
        # The elements outside [i, j] equal to k remain k.
        # The elements inside [i, j] equal to v become k.
        # Elements inside [i, j] not equal to v remain not k (since nums[m] != v implies nums[m] + x != k).
        # Elements outside [i, j] not equal to k remain not k.
        # The total frequency of k becomes:
        # (Count of k outside [i, j]) + (Count of v inside [i, j])
        # = (Total count of k) - (Count of k inside [i, j]) + (Count of v inside [i, j])
        # = total_k + (Count of v inside [i, j] - Count of k inside [i, j])
        # We want to maximize the term (Count of v inside [i, j] - Count of k inside [i, j])
        # over all v in distinct_values and all subarrays [i, j].
        # This term is exactly the sum of a virtual array 'w' over the subarray [i, j],
        # where w[m] = 1 if nums[m] == v and nums[m] != k,
        # w[m] = -1 if nums[m] == k and nums[m] != v,
        # w[m] = 0 otherwise (when nums[m] == v == k or nums[m] != v and nums[m] != k).
        # We can use Kadane's algorithm to find the maximum subarray sum for this 'w' array.

        for v in distinct_values:
             # Construct the first element w_0 based on the refined definition
             if nums[0] == v and nums[0] != k:
                 w_0 = 1
             elif nums[0] == k and nums[0] != v:
                 w_0 = -1
             else: # nums[0] == v == k or nums[0] != v and nums[0] != k
                 w_0 = 0

             # Initialize Kadane's variables for the current v
             # max_so_far stores the maximum subarray sum found ending at any position.
             # current_max stores the maximum subarray sum found ending at the current position.
             # We need the standard Kadane's that can return negative values,
             # as the maximum possible change (max_overall_kadane_sum) could be negative.
             current_max = w_0
             max_so_far = w_0 # Max sum of any non-empty subarray ending so far.

             # Apply Kadane's algorithm for the virtual array 'w'
             for m in range(1, n):
                 # Construct the current element w_m based on the refined definition
                 if nums[m] == v and nums[m] != k:
                     w_m = 1
                 elif nums[m] == k and nums[m] != v:
                     w_m = -1
                 else: # nums[m] == v == k or nums[m] != v and nums[m] != k
                     w_m = 0

                 # Standard Kadane's update
                 # current_max is the maximum of starting a new subarray at m (value w_m)
                 # or extending the current subarray (current_max + w_m).
                 current_max = max(w_m, current_max + w_m)
                 # max_so_far is the maximum sum found over any non-empty subarray ending so far.
                 max_so_far = max(max_so_far, current_max)

             # After Kadane's for the current v, max_so_far is the max sum of any non-empty
             # subarray of w. This represents the max change N_v(i,j) - N_k(i,j) for this v
             # over all non-empty subarrays [i,j].
             current_kadane_sum = max_so_far

             # Update the overall maximum Kadane sum found across all values of v.
             max_overall_kadane_sum = max(max_overall_kadane_sum, current_kadane_sum)

        # Step 4: Calculate the maximum frequency
        # The maximum frequency is total_k (initial frequency from elements outside the modified subarray)
        # plus the maximum possible increase achieved by modifying a subarray.
        # The maximum possible increase within the subarray is max_overall_kadane_sum
        # (the max of N_v(i,j) - N_k(i,j) over all v and non-empty [i,j]).
        # However, we can always choose an empty subarray modification (i > j), which results
        # in a change of 0 (N_v(i,j)=0, N_k(i,j)=0). If max_overall_kadane_sum is negative,
        # it means modifying any non-empty subarray corresponding to any v reduces the count of k
        # or increases it by less than the count of k removed from the subarray. In such cases,
        # the best frequency is the original frequency total_k (achieved by an empty subarray,
        # which gives a change of 0).
        max_increase = max(0, max_overall_kadane_sum)

        return total_k + max_increase