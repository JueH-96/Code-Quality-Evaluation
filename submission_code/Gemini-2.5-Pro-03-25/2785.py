from typing import List

class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of adjacent swaps to make the permutation semi-ordered.
        A semi-ordered permutation has 1 at the first position (index 0) and n at the last position (index n-1).
        The minimum number of adjacent swaps required to move an element to a target position is equal to the distance
        between its current position and the target position.

        Args:
            nums: A 0-indexed permutation of n integers, where n is the length of nums.

        Returns:
            The minimum number of operations (adjacent swaps) required to make the permutation semi-ordered.
        """
        n = len(nums)
        # Constraints state 2 <= n. This implies n is at least 2, and therefore 1 and n are distinct elements.

        idx1 = -1  # Initialize index of the element 1
        idxn = -1  # Initialize index of the element n

        # Find the 0-based indices of the number 1 and the number n in the list.
        # We iterate through the list to find these indices.
        # Since nums is a permutation of integers from 1 to n, both 1 and n are guaranteed to be present exactly once.
        for i in range(n):
            if nums[i] == 1:
                idx1 = i
            elif nums[i] == n: # Using elif is safe because an element cannot be both 1 and n (since n >= 2)
                idxn = i

            # Optimization: If we have found both indices, we can stop searching early
            # as we only need the positions of 1 and n.
            if idx1 != -1 and idxn != -1:
                break

        # Calculate the number of swaps needed to move the element 1 from its current position (idx1)
        # to the target position (index 0). This requires moving it idx1 steps to the left,
        # which takes exactly idx1 adjacent swaps.
        swaps_for_1 = idx1

        # Calculate the number of swaps needed to move the element n from its current position (idxn)
        # to the target position (index n-1). This requires moving it (n - 1) - idxn steps to the right,
        # which takes exactly (n - 1) - idxn adjacent swaps.
        swaps_for_n = (n - 1) - idxn

        # The total number of swaps seems to be the sum of swaps required for 1 and n independently.
        # Let's compute this sum first.
        total_swaps = swaps_for_1 + swaps_for_n

        # Now, we need to consider the interaction between the movements of 1 and n.
        # The key insight is about whether the path of 1 (moving left) crosses the path of n (moving right).
        # This happens if and only if 1 starts to the right of n (i.e., idx1 > idxn).

        # Case 1: idx1 < idxn (1 is initially to the left of n)
        # When 1 moves left and n moves right, their paths do not cross. The swaps required
        # to move 1 are independent of the swaps required to move n.
        # The total minimum swaps is indeed the sum: swaps_for_1 + swaps_for_n.

        # Case 2: idx1 > idxn (1 is initially to the right of n)
        # When 1 moves left and n moves right, their paths must cross. At some point,
        # 1 and n will become adjacent, like [... n, 1 ...]. The next swap will be between
        # n and 1, resulting in [... 1, n ...]. This single swap operation achieves two things:
        # it moves 1 one step closer to index 0, and it moves n one step closer to index n-1.
        # In our initial calculation (swaps_for_1 + swaps_for_n), we counted the required number
        # of steps for 1 (idx1) and the required number of steps for n ((n-1)-idxn) independently.
        # However, the single swap involving both 1 and n contributes one step to both counts.
        # Thus, the sum `swaps_for_1 + swaps_for_n` overcounts the total number of swaps by exactly 1
        # in this case, because one swap does double duty.
        # Therefore, we need to subtract 1 from the sum to get the correct minimum number of swaps.
        if idx1 > idxn:
            total_swaps -= 1

        # The final result is the calculated total_swaps.
        return total_swaps