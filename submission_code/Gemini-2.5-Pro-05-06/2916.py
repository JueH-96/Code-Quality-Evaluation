from typing import List

class Solution:
  def canSplitArray(self, nums: List[int], m: int) -> bool:
    n = len(nums)

    # Constraints: 1 <= n <= 100. nums is never empty.
    # If n=1, the array is already "split" into 1 array of length 1.
    # The DP logic below correctly handles n=1:
    #   - The 'length' loop range(2, 1+1) = range(2,2) is empty.
    #   - dp[0][0] is initialized to True (base case).
    #   - The function returns dp[0][n-1] which is dp[0][0] = True.
    # So, no special 'if n == 1:' check is explicitly needed here.

    # Precompute prefix sums for O(1) sum queries of subarrays.
    # prefix_sums[k] stores sum of nums[0...k-1]. prefix_sums[0] = 0.
    # So, sum(nums[i...j]) = prefix_sums[j+1] - prefix_sums[i].
    prefix_sums = [0] * (n + 1)
    for i in range(n):
        prefix_sums[i+1] = prefix_sums[i] + nums[i]

    # dp[i][j] will be True if the subarray nums[i...j] (inclusive)
    # can be completely split into j-i+1 arrays of length 1,
    # according to the problem's rules.
    # Initialize dp table with False.
    dp = [[False for _ in range(n)] for _ in range(n)]

    # Base cases: Subarrays of length 1 (i.e., dp[i][i]) are intrinsically "split".
    for i in range(n):
        dp[i][i] = True

    # Fill DP table. Iterate over subarray lengths, from L=2 up to L=n.
    for length in range(2, n + 1):
        # Iterate over all possible starting indices 'i' for subarrays of current 'length'.
        for i in range(n - length + 1):
            j = i + length - 1  # Ending index of the current subarray nums[i...j]
            
            # Try all possible split points 'k'.
            # The subarray nums[i...j] is split into:
            # Left part: nums[i...k]
            # Right part: nums[k+1...j]
            # 'k' ranges from 'i' (left part is nums[i]) to 'j-1' (right part is nums[j]).
            for k in range(i, j):
                # Properties of the left part (nums[i...k])
                sum_left_part = prefix_sums[k+1] - prefix_sums[i]
                len_left_part = k - i + 1
                
                # Properties of the right part (nums[k+1...j])
                sum_right_part = prefix_sums[j+1] - prefix_sums[k+1]
                len_right_part = j - k  # This is (j - (k+1) + 1)

                # Check if this specific split is valid based on problem rules.
                # For each of the two resulting subarrays (left and right),
                # its length must be 1 OR its sum must be >= m.
                
                is_left_part_condition_met = (len_left_part == 1) or (sum_left_part >= m)
                is_right_part_condition_met = (len_right_part == 1) or (sum_right_part >= m)

                if is_left_part_condition_met and is_right_part_condition_met:
                    # If the current split (dividing nums[i...j] into nums[i...k] and nums[k+1...j]) is valid,
                    # then nums[i...j] can be fully split if BOTH nums[i...k] AND nums[k+1...j]
                    # can themselves be further split down to individual elements.
                    # This recursive requirement is captured by dp[i][k] and dp[k+1][j].
                    if dp[i][k] and dp[k+1][j]:
                        dp[i][j] = True
                        # Found a valid way to split nums[i...j],
                        # no need to check other split points 'k' for this [i,j].
                        break 
            
    # The final answer is whether the entire array nums[0...n-1] can be split.
    return dp[0][n-1]