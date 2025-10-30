from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        """
        Finds a non-empty subsequence with alternating sum k and maximized product <= limit.
        """
        # dp_even[s] = p means there is a subsequence of even length 
        # with alternating sum s and max product p.
        # dp_odd[s] = p means there is a subsequence of odd length 
        # with alternating sum s and max product p.
        dp_even = {}
        dp_odd = {}

        for num in nums:
            # Create copies to hold the DP states for the current iteration.
            # This prevents using `num` with subsequences already formed in this same iteration.
            next_dp_even = dp_even.copy()
            next_dp_odd = dp_odd.copy()

            # Case 1: Start a new subsequence `[num]`.
            # This has odd length (1). Alternating sum is `num`, product is `num`.
            if num <= limit:
                # Update the max product for this sum in the odd-length table.
                current_max_prod = next_dp_odd.get(num, -1)
                next_dp_odd[num] = max(current_max_prod, num)

            # Case 2: Append `num` to existing subsequences.
            
            # Append `num` to subsequences of even length.
            # The new element `num` is at an even index (0-indexed), so it's added.
            # The new subsequence has an odd length.
            for s, p in dp_even.items():
                new_s = s + num
                new_p = p * num
                if new_p <= limit:
                    current_max_prod = next_dp_odd.get(new_s, -1)
                    next_dp_odd[new_s] = max(current_max_prod, new_p)

            # Append `num` to subsequences of odd length.
            # The new element `num` is at an odd index, so it's subtracted.
            # The new subsequence has an even length.
            for s, p in dp_odd.items():
                new_s = s - num
                new_p = p * num
                if new_p <= limit:
                    current_max_prod = next_dp_even.get(new_s, -1)
                    next_dp_even[new_s] = max(current_max_prod, new_p)
            
            # Update the DP tables for the next number in nums.
            dp_even = next_dp_even
            dp_odd = next_dp_odd

        # The final subsequence can have either even or odd length.
        # We check both tables for the target sum k.
        # .get(k, -1) handles cases where no such subsequence exists.
        res_even = dp_even.get(k, -1)
        res_odd = dp_odd.get(k, -1)
        
        return max(res_even, res_odd)