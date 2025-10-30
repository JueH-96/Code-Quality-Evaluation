from typing import List
import collections

class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # Collect unique values in nums + k as potential target original values (v)
        # An element nums[p] inside the subarray [i..j] becomes k if nums[p] + x = k,
        # meaning nums[p] must be k - x. If we want multiple elements inside [i..j]
        # to become k using the same x, they must have the same original value v = k - x.
        # This value v must be one of the original values present in nums[i..j].
        # To maximize the count, we can iterate through all unique values present in
        # the original array nums, as potential candidates for v. The value k itself
        # is also a candidate for v (if x=0 is used).
        unique_nums_values = set(nums)
        
        # possible_v_set contains all unique values from nums + k.
        possible_v_set = unique_nums_values | {k} # Union of sets
        
        # Precompute total count of k in original nums
        total_k_count = 0
        for x in nums:
            if x == k:
                total_k_count += 1
                
        # Maximum frequency is at least the original frequency of k.
        # This can be achieved by selecting an empty subarray (i > j),
        # or by adding x=0 to a subarray that doesn't affect any existing k's negatively.
        # The minimum possible frequency is total_k_count.
        overall_max_freq = total_k_count
        
        # Precompute prefix sums for k once
        # count_k_prefix[i] = number of times k appears in nums[0..i]
        count_k_prefix = [0] * n
        current_count_k = 0
        for i in range(n):
            if nums[i] == k:
                current_count_k += 1
            count_k_prefix[i] = current_count_k

        # Iterate through each possible original value 'v' that we transform into 'k'
        for v in possible_v_set:
            # If v == k, adding x=0 doesn't change anything. The frequency is just the
            # total count of k in the original array. Our calculation below handles
            # this case correctly (diff_vk will be all zeros, max_subarray_diff = 0).

            # Compute prefix sums for v on the fly
            # count_v_prefix[i] = number of times v appears in nums[0..i]
            count_v_prefix = [0] * n
            current_count_v = 0
            for i in range(n):
                if nums[i] == v:
                    current_count_v += 1
                count_v_prefix[i] = current_count_v

            # We want to maximize:
            # Count(original v in nums[i..j]) + Count(original k outside nums[i..j])
            # = (Count(v in nums[0..j]) - Count(v in nums[0..i-1])) + (Count(k in nums[0..n-1]) - Count(k in nums[i..j]))
            # = (Count(v in nums[0..j]) - Count(v in nums[0..i-1])) + (Total k count - (Count(k in nums[0..j]) - Count(k in nums[0..i-1])))
            # = (Count(v in nums[0..j]) - Count(k in nums[0..j])) - (Count(v in nums[0..i-1]) - Count(k in nums[0..i-1])) + Total k count

            # Let diff_vk[p] = Count(v in nums[0..p]) - Count(k in nums[0..p]) for p = 0..n-1.
            # Let diff_vk[-1] = 0.
            # We want to maximize diff_vk[j] - diff_vk[i-1] + Total k count over 0 <= i <= j <= n-1.
            # This is equivalent to maximizing diff_vk[j] - diff_vk[i-1] over -1 <= i-1 < j <= n-1.

            # Let diff_vk_padded be an array of size n+1 where:
            # diff_vk_padded[0] = diff_vk[-1] = 0
            # diff_vk_padded[p+1] = diff_vk[p] for p = 0..n-1
            # We want to maximize diff_vk_padded[idx2] - diff_vk_padded[idx1] over 0 <= idx1 < idx2 <= n.

            diff_vk_padded = [0] # diff_vk_padded[0] = diff_vk[-1] = 0
            for p in range(n):
                diff_vk_padded.append(count_v_prefix[p] - count_k_prefix[p])

            # Find max_subarray_diff = max_{0 <= idx1 < idx2 <= n} (diff_vk_padded[idx2] - diff_vk_padded[idx1])
            # This represents the maximum possible value of (Count(v in nums[i..j]) - Count(k in nums[i..j]))

            # Use Kadane's-like approach to find maximum difference (p2 > p1)
            # max_subarray_diff initialized to 0 covers the case where the maximum
            # difference is non-positive, or achieved with i=j.
            max_subarray_diff = 0 
            
            # min_prefix stores the minimum value of diff_vk_padded encountered so far.
            # Initialize with the first value diff_vk_padded[0] (corresponding to index i-1 = -1).
            min_prefix = diff_vk_padded[0]

            # Iterate from the second value onwards (corresponding to original index p=0, i=0)
            # idx2 goes from 1 to n.
            for idx2 in range(1, n + 1):
                # The difference is diff_vk_padded[idx2] minus the minimum value seen at or before idx2-1.
                # By updating min_prefix *after* calculating the difference,
                # min_prefix always stores the minimum value up to the previous index (idx2-1).
                max_subarray_diff = max(max_subarray_diff, diff_vk_padded[idx2] - min_prefix)
                
                # Update min_prefix encountered so far (at indices <= idx2)
                min_prefix = min(min_prefix, diff_vk_padded[idx2])

            # The maximum frequency for this 'v' is max_subarray_diff + total_k_count
            # max_subarray_diff = max_{0 <= i <= j <= n-1} (Count(v in [i..j]) - Count(k in [i..j]))
            # The max frequency is max_{0 <= i <= j <= n-1} (Count(v in [i..j]) - Count(k in [i..j]) + Count(k in [0..n-1]))
            # which simplifies to max_{0 <= i <= j <= n-1} (Count(v in [i..j]) + Count(k outside [i..j]))
            current_max_freq = max_subarray_diff + total_k_count
            
            overall_max_freq = max(overall_max_freq, current_max_freq)
        
        return overall_max_freq