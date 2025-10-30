import collections
from typing import List

class Solution:
  """
  Finds the maximum length of a "good" subsequence.
  A subsequence `seq` is good if the number of indices `i` such that `seq[i] != seq[i+1]` is at most `k`.
  Uses dynamic programming with optimization.
  """
  def maximumLength(self, nums: List[int], k: int) -> int:
    """
    Calculates the maximum length of a good subsequence using dynamic programming.

    The state `dp[j][val]` will store the maximum length of a good subsequence ending with value `val` 
    that has used exactly `j` differences (i.e., exactly `j` indices `i` where `seq[i] != seq[i+1]`).

    To efficiently compute the state transitions, we maintain overall maximum lengths achieved for each 
    difference count `j`. Specifically, we track the top 1 and top 2 maximum lengths (`max_len1[j]`, `max_len2[j]`) 
    and the values they end with (`val1[j]`, `val2[j]`). This allows O(1) lookup for finding the maximum length 
    of a subsequence ending with a different value when extending a sequence.

    Time complexity: O(N * K), where N is the length of nums. We iterate through each number in nums, 
                     and for each number, we iterate through k+1 possible difference counts.
    Space complexity: O(D * K), where D is the number of distinct values in nums. 
                      In the worst case, D can be up to N, making the space complexity O(N * K).
                      This is dominated by the `dp` dictionaries. The `max_len` and `val` arrays take O(K) space.
    """
    n = len(nums)
    if n == 0:
        return 0

    # dp[j][val] stores the maximum length of a subsequence ending with value `val`
    # using exactly `j` differences.
    # We use a list of dictionaries: dp[j] is a defaultdict mapping val -> max_len.
    # collections.defaultdict(int) automatically initializes missing keys with 0.
    dp = [collections.defaultdict(int) for _ in range(k + 1)]

    # Keep track of the top 2 maximum lengths achieved for each difference count `j` across all values,
    # and their corresponding ending values. This optimization avoids iterating through all previous values.
    # max_len1[j] = max length using exactly j differences.
    # val1[j] = the value the max_len1[j] sequence ends with.
    # max_len2[j] = second max length using exactly j differences.
    # val2[j] = the value the max_len2[j] sequence ends with.
    max_len1 = [0] * (k + 1)  
    val1 = [None] * (k + 1)   
    max_len2 = [0] * (k + 1) 
    val2 = [None] * (k + 1)   

    # Iterate through each number in the input array `nums`.
    for i in range(n):
        current_val = nums[i]
        
        # Temporary storage for computed max lengths ending at nums[i] for each difference count j.
        # We compute these lengths based on the state *before* processing nums[i].
        # We iterate j downwards (from k to 0) to ensure that when calculating for difference j,
        # the values for j-1 (needed for len2 calculation) are from the previous step (i-1), not partially updated from the current step i.
        prev_max_len_for_curr_i = [0] * (k + 1) 

        for j in range(k, -1, -1):
            # Calculate potential max length ending at nums[i] with exactly j differences.

            # Option 1: Extend a subsequence ending with current_val. 
            # The number of differences `j` remains the same.
            # Length = (previous max length ending with current_val with j diffs) + 1
            len1 = dp[j][current_val] + 1

            # Option 2: Extend a subsequence ending with prev_val != current_val. 
            # This uses one more difference, so the previous sequence must have used j-1 differences.
            # Requires j > 0.
            # Length = (max length ending with any prev_val!=current_val using j-1 diffs) + 1.
            len2 = 0
            if j > 0:
                max_prev_len_diff_val = 0
                # Use the tracked top 1 and top 2 lengths for difference count j-1 to find the max length
                # ending in a value different from current_val.
                if max_len1[j - 1] > 0: # Check if any sequence exists with j-1 differences
                    if val1[j - 1] != current_val:
                        # If the globally best sequence for j-1 differences doesn't end with current_val, we can use its length.
                        max_prev_len_diff_val = max_len1[j - 1]
                    else:  
                        # If the globally best sequence ends with current_val, we must use the second best sequence's length.
                        max_prev_len_diff_val = max_len2[j - 1]
                
                # If a valid previous sequence (length > 0) ending in a different value is found, calculate the new length.
                if max_prev_len_diff_val > 0: 
                   len2 = max_prev_len_diff_val + 1

            # Determine the maximum length ending at nums[i] with exactly j differences.
            if j == 0:
                 # For 0 differences, the sequence must consist of identical elements equal to current_val.
                 # The base case subsequence [nums[i]] has length 1.
                 # We take the maximum of extending an existing sequence of `current_val`s or starting a new one of length 1.
                 current_max_len_j = max(1, len1) 
            else:
                 # For j > 0 differences, consider both extension options.
                 current_max_len_j = max(len1, len2)
                 # If len1 and len2 are both 0, means no sequence can end at nums[i] with exactly j diffs.
                 # If len1 comes from dp[j][current_val]=0, len1=1. If len2 comes from max_prev=0, len2=0. Max is 1. This seems right.

            # Store the computed length temporarily before updating the main state.
            prev_max_len_for_curr_i[j] = current_max_len_j

        # After computing potential max lengths for all j for the current element nums[i],
        # update the main dp state and the overall max length trackers (max_len1, max_len2).
        for j in range(k + 1):
             new_len = prev_max_len_for_curr_i[j]
             
             # Update the dp state for current_val and difference count j only if we found a longer sequence.
             # This check is crucial because updating overall max trackers should only happen when
             # a strictly better path for this specific (value, j) combination is discovered.
             if new_len > dp[j][current_val]:
                dp[j][current_val] = new_len

                # Check if this new length updates the overall top 1 or top 2 max lengths for this difference count j.
                if new_len > max_len1[j]:
                    # If the new length is better than the current global best for j differences:
                    if current_val == val1[j]:  
                        # If it's the same value as the current best, just update the length.
                        max_len1[j] = new_len
                    else:  
                        # If it's a new value, it becomes the best. The old best becomes the second best.
                        max_len2[j] = max_len1[j]
                        val2[j] = val1[j]
                        max_len1[j] = new_len
                        val1[j] = current_val
                elif new_len > max_len2[j]:
                     # If the new length is between the current top 1 and top 2 lengths:
                     # Check if this value is different from the current top value. If so, it becomes the new second best.
                    if current_val != val1[j]:
                        max_len2[j] = new_len
                        val2[j] = current_val

    # The problem asks for the maximum length of a subsequence with *at most* k differences.
    # Since max_len1[j] stores the maximum length achieved using *exactly* j differences,
    # the final answer is the maximum value among max_len1[0], max_len1[1], ..., max_len1[k].
    final_max_len = 0
    for length in max_len1:
        final_max_len = max(final_max_len, length)
            
    # The logic ensures final_max_len >= 1 if n > 0, because max_len1[0] is updated correctly.
    return final_max_len