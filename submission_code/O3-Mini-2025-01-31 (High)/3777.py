from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # We'll use dynamic programming where each state is represented by a pair:
        #   (alternating_sum, parity)
        #
        # "alternating_sum" is the current alternating sum of the chosen subsequence,
        # and "parity" indicates the number of elements in the subsequence mod 2.
        # The idea is that when you add a new number:
        #   • If the current subsequence length is even (p == 0), then the new number
        #     will be appended at an even index (0-indexed) and will be added.
        #   • If the current subsequence length is odd (p == 1), then the new number
        #     goes to an odd index and will be subtracted.
        #
        # In addition, for every state we store the maximum product achieved (which is
        # always <= limit). We do not allow the empty subsequence.
        #
        # Transition:
        #   From a state (alt, p) with product P, when choosing a candidate number x:
        #      - If p == 0, new state becomes (alt + x, 1)
        #      - If p == 1, new state becomes (alt - x, 0)
        #   The new product is P * x (only if P * x <= limit).
        #
        # Also, at any position we may start a new subsequence using the current number.
        
        # dp: dictionary mapping (alternating_sum, parity) -> maximum product so far.
        dp = {}
        
        for num in nums:
            # We'll update the dp using this number.
            new_dp = dp.copy()  # states from previous indices are always allowed (skip option)
            
            # Extend every existing subsequence with the current number.
            # Note: iterate over a list of items to avoid updating while iterating.
            for (alt, p), prod in list(dp.items()):
                new_product = prod * num
                if new_product <= limit:
                    if p == 0:  # next element goes to an even index, so add num
                        new_state = (alt + num, 1)
                    else:       # next element goes to an odd index, so subtract num
                        new_state = (alt - num, 0)
                    # We want the maximum product for each state.
                    if new_product > new_dp.get(new_state, -1):
                        new_dp[new_state] = new_product
            
            # Also, we can start a new subsequence with the current number,
            # which will be the first element (thus, added to the sum).
            if num <= limit:
                new_state = (num, 1)  # After one element, subsequence length = 1 (parity 1)
                if num > new_dp.get(new_state, -1):
                    new_dp[new_state] = num
            
            dp = new_dp

        # Finally, among all states with alternating sum exactly equal to k,
        # we pick the one with the maximum product.
        ans = -1
        for (alt, _), prod in dp.items():
            if alt == k:
                ans = max(ans, prod)
        return ans