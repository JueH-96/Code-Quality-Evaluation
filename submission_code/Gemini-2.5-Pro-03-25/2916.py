import sys 
# sys.setrecursionlimit(2000) # Adjust recursion depth if needed for recursive solutions
from typing import List

class Solution:
    """
    You are given an array nums of length n and an integer m. You need to determine if it is possible to split the array into n non-empty arrays by performing a series of steps.
    In each step, you can select an existing array (which may be the result of previous steps) with a length of at least two and split it into two subarrays, if, for each resulting subarray, at least one of the following holds:

    1. The length of the subarray is one, or
    2. The sum of elements of the subarray is greater than or equal to m.

    Return true if you can split the given array into n arrays (each of length 1), otherwise return false.
    Note: A subarray is a contiguous non-empty sequence of elements within an array.
    """
    def canSplitArray(self, nums: List[int], m: int) -> bool:
        """
        Determines if the array can be split into n non-empty arrays
        following the given splitting rules.

        Args:
            nums: The list of integers.
            m: The integer threshold for subarray sums.

        Returns:
            True if the array can be split into n arrays of length 1, False otherwise.
        """
        n = len(nums)

        # Base cases:
        # If n=1, the array is already split into 1 array (the target state). No operations are needed or possible.
        # Thus, it's possible to achieve the state of n=1 array of length 1.
        # If n=2, the array [a, b] needs one split into [a] and [b].
        #   The split condition requires checking the two resulting subarrays: [a] and [b].
        #   Condition for a subarray `sub`: C(sub) = (len(sub) == 1 or sum(sub) >= m).
        #   For [a]: C([a]) = (len([a]) == 1 or sum([a]) >= m) = (True or a >= m) = True.
        #   For [b]: C([b]) = (len([b]) == 1 or sum([b]) >= m) = (True or b >= m) = True.
        #   Since C([a]) and C([b]) are both true, the split is always valid for n=2.
        if n <= 2:
            return True

        # For n > 2:
        # The splitting process requires n-1 valid splits in total.
        # The final splits will necessarily involve splitting arrays of length 2. As shown above,
        # splitting any array of length 2 is always valid according to the rule C.
        #
        # The constraints arise when splitting arrays of length greater than 2.
        # When splitting an array A into L and R, the split is valid if C(L) and C(R) both hold.
        # If len(L) > 1, the condition C(L) simplifies to sum(L) >= m.
        # If len(R) > 1, the condition C(R) simplifies to sum(R) >= m.
        #
        # Consider the case n=3: nums = [a, b, c]. We need to perform the first split.
        # Option 1: Split into L=[a], R=[b, c]. Valid if C([a]) and C([b, c]) hold.
        #   C([a]) is True. C([b, c]) = (len=2 or sum=b+c >= m). Requires b+c >= m.
        #   If this split is valid (i.e., b+c >= m), we then need to split [b, c], which is always possible.
        # Option 2: Split into L=[a, b], R=[c]. Valid if C([a, b]) and C([c]) hold.
        #   C([c]) is True. C([a, b]) = (len=2 or sum=a+b >= m). Requires a+b >= m.
        #   If this split is valid (i.e., a+b >= m), we then need to split [a, b], which is always possible.
        # Therefore, for n=3, the array can be split iff (a+b >= m) or (b+c >= m).
        #
        # This observation suggests a hypothesis: for n > 2, the array can be split if and only if
        # there exists at least one pair of adjacent elements nums[i], nums[i+1] such that their sum is >= m.
        # If such a pair exists, it seems to guarantee that a valid sequence of splits can be constructed.
        # The logic is that this condition provides a necessary "seed" for satisfying the sum constraints
        # for subarrays of length > 1 created during the process.
        # Conversely, if no adjacent pair sums to >= m, it appears impossible to satisfy the split conditions
        # for n > 2, as demonstrated by the n=3 case. This pattern seems generalizable.

        # We check if any adjacent pair satisfies the condition sum >= m.
        # Iterate through all adjacent pairs in the array (from index 0 to n-2).
        for i in range(n - 1):
            # Calculate the sum of the current adjacent pair.
            adjacent_sum = nums[i] + nums[i+1]
            
            # Check if the sum meets the threshold m.
            if adjacent_sum >= m:
                # If we find such a pair, according to the hypothesis, it's possible to split the array.
                return True

        # If the loop completes without finding any adjacent pair whose sum is >= m,
        # then according to the hypothesis, it's impossible to split the array when n > 2.
        return False