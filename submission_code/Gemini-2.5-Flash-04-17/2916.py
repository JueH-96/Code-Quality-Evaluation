from typing import List

class Solution:
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        n = len(nums)
        
        # If n is 1 or 2, the array can always be considered split into n singletons.
        # If n=1, it's already a single array of length 1. We need to form n=1 non-empty array. This is possible.
        # If n=2, [a, b] can be split into [a] and [b]. Both resulting subarrays have length 1.
        # The split rule says for each resulting subarray, at least one of the following holds:
        # 1. The length of the subarray is one, or
        # 2. The sum of elements of the subarray is greater than or equal to m.
        # Since both [a] and [b] have length 1, condition 1 is met for both.
        # Thus, splitting an array of length 2 into two length-1 arrays is always a valid split.
        # Therefore, for n=1 or n=2, it's always possible to reach the state of n singletons.
        if n <= 2:
            return True
        
        # If n > 2, we need to perform at least one split.
        # Consider a subarray `arr` of length L >= 2 that we want to split into `arr_left` (length L1 >= 1)
        # and `arr_right` (length L2 >= 1), where L1 + L2 = L.
        # The split is valid if (L1 == 1 OR sum(arr_left) >= m) AND (L2 == 1 OR sum(arr_right) >= m).
        #
        # If L = 3, arr = [a, b, c].
        # Possible splits:
        # 1. Split into [a] (L1=1) and [b, c] (L2=2). Valid iff (true OR sum([a])>=m) AND (false OR sum([b, c])>=m).
        #    This simplifies to true AND (sum([b, c])>=m) if sum([b, c]) >= m. Valid iff b+c >= m.
        # 2. Split into [a, b] (L1=2) and [c] (L2=1). Valid iff (false OR sum([a, b])>=m) AND (true OR sum([c])>=m).
        #    This simplifies to (sum([a, b])>=m) AND true if sum([a, b]) >= m. Valid iff a+b >= m.
        #
        # So, a length-3 subarray [a, b, c] can be validly split into smaller pieces (which can then be further split into singletons as length-2 pieces can always be split)
        # if and only if a+b >= m OR b+c >= m.
        # If neither a+b >= m nor b+c >= m, a length-3 subarray cannot be split at all according to the rules.
        #
        # If the original array has length n > 2 and contains no adjacent pair nums[i], nums[i+1] with sum >= m,
        # then any subarray of length 3, say nums[i...i+2], will have nums[i]+nums[i+1] < m AND nums[i+1]+nums[i+2] < m.
        # As shown above, such a length-3 subarray cannot be split.
        # If n=3 and both adjacent sums < m, the original array cannot be split.
        # If n>3 and all adjacent sums < m, it seems impossible to avoid needing to split a length >= 3 subarray,
        # and any length >= 3 subarray will eventually need to be broken down. If it cannot be broken down
        # into valid smaller pieces (like the length-3 case where adjacent sums are < m), the process fails.
        # Conversely, if there is at least one adjacent pair nums[i], nums[i+1] with sum >= m,
        # it provides a "pivot" or mechanism that seems to allow the array to be fully split.
        # The hypothesis is that for n > 2, it's possible if and only if such an adjacent pair exists.
        
        # Check if there exists an adjacent pair with sum >= m
        for i in range(n - 1):
            if nums[i] + nums[i+1] >= m:
                # Found an adjacent pair with sum >= m, so it is possible according to the hypothesis.
                return True
                
        # If no such adjacent pair exists after checking all adjacent pairs,
        # and n > 2, it is impossible to split into n singletons according to the hypothesis.
        return False