import collections
from typing import List

class Solution:
  def maximumLength(self, nums: List[int], k: int) -> int:
    """
    Finds the maximum possible length of a "good" subsequence.
    A subsequence is good if there are at most k indices i such that seq[i] != seq[i+1].
    """
    n = len(nums)
    if n == 0:
        return 0

    # f[num][j]: max length of a subsequence ending in `num` with exactly `j` breaks.
    f = collections.defaultdict(lambda: [0] * (k + 1))
    
    # dp[j] stores the top two lengths for subsequences with j breaks,
    # along with the numbers they end with.
    # Format: dp[j] = [(max_len, num), (second_max_len, num)]
    dp = [[(0, -1), (0, -1)] for _ in range(k + 1)]
    
    ans = 0

    for num in nums:
        # We need to read all old values before writing new ones for this `num`.
        # So, calculate new lengths for `num` based on the state *before* this num.
        new_f_num = [0] * (k + 1)
        
        for j in range(k + 1):
            # Case 1: Extend a subsequence that already ends with `num`.
            # Breaks do not increase.
            len_same = 1 + f[num][j]

            # Case 2: Extend a subsequence that ends with a different number.
            # Breaks increase by 1, so we need a subsequence with j-1 breaks.
            len_diff = 0
            if j > 0:
                max1_len, max1_num = dp[j - 1][0]
                max2_len, _ = dp[j - 1][1]
                
                if max1_num != num:
                    len_diff = 1 + max1_len
                else:  # The best one ends in num, so we must use the second best.
                    len_diff = 1 + max2_len

            # The new max length for a subsequence ending in `num` with `j` breaks.
            current_len = max(len_same, len_diff)
            new_f_num[j] = current_len
        
        # Update f[num] with the newly calculated lengths.
        f[num] = new_f_num

        # Update the global top-two tracker `dp` and the overall answer.
        for j in range(k + 1):
            new_len = f[num][j]
            ans = max(ans, new_len)
            
            max1_len, max1_num = dp[j][0]
            max2_len, _ = dp[j][1]

            if new_len > max1_len:
                if num != max1_num:
                    # old max1 becomes new max2
                    dp[j][1] = dp[j][0]
                # new_len is the new max1
                dp[j][0] = (new_len, num)
            elif new_len > max2_len:
                if num != max1_num:
                    dp[j][1] = (new_len, num)
                    
    return ans