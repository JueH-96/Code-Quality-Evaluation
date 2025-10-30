from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # We'll use dynamic programming where the state is a tuple
        # (parity, alt, used) where:
        #   - parity indicates the parity of the current subsequence length (i.e. count mod 2).
        #     If parity == 0, it means we have picked an even number of elements,
        #     so the next picked element will be at an even index (and will be added).
        #     If parity == 1, the next picked element will be at an odd index (and will be subtracted).
        #   - alt is the alternating sum of the subsequence so far.
        #   - used is a boolean flag indicating whether we've chosen at least one element
        #     (because the empty subsequence is not allowed).
        #
        # For each state, we store the maximum product that can be achieved for that state,
        # with the invariant that the product never exceeds limit.
        #
        # Initialize dp with the "empty state":
        # (0, 0, False) with product 1. (We use product=1 as the multiplicative identity.)
        #
        # Then, process each number in nums in order. For each number x,
        # we have two choices: skip it (state remains unchanged) or take it.
        # When taking it, the effect depends on the current parity:
        #  - If parity == 0, then this position is even, the alternating sum increases by x.
        #  - If parity == 1, then this position is odd, the alternating sum decreases by x.
        # and the new parity becomes 1 - current parity.
        # The new product becomes old_product * x.
        # We only allow this transition if the new product is <= limit.
        #
        # Finally, among all states with used==True and alt == k we pick the maximum product.
        # If no such state exists, return -1.
        
        # dp is a dictionary: key: (parity, alt, used)   value: maximum product (an int)
        dp = {(0, 0, False): 1}
        
        for x in nums:
            # We will update dp to newdp in this iteration.
            newdp = dict(dp)  # states from "not picking" remain the same.
            for (parity, alt, used), prod in dp.items():
                # When we decide to pick the current element x:
                new_prod = prod * x
                if new_prod > limit:
                    # skip transition if product exceeds limit.
                    continue
                # Determine the effect on alternating sum.
                # When a new element is added:
                # - If the current parity is 0 (even count so far), new element is in even position -> add.
                # - If the current parity is 1, new element is in odd position -> subtract.
                if parity == 0:
                    new_alt = alt + x
                else:
                    new_alt = alt - x
                new_parity = 1 - parity
                # Mark that we've used at least one element.
                new_used = True
                # In the new dp state (new_parity, new_alt, new_used), we want the maximum product.
                key = (new_parity, new_alt, new_used)
                if key in newdp:
                    if new_prod > newdp[key]:
                        newdp[key] = new_prod
                else:
                    newdp[key] = new_prod
            dp = newdp
        
        # After processing all numbers, look for any state with used==True and alt==k.
        ans = -1
        for (parity, alt, used), prod in dp.items():
            if used and alt == k:
                ans = max(ans, prod)
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maxProduct([1, 2, 3], 2, 10))  # Expected output: 6
    # Example 2:
    print(sol.maxProduct([0, 2, 3], -5, 12))  # Expected output: -1
    # Example 3:
    print(sol.maxProduct([2, 2, 3, 3], 0, 9))  # Expected output: 9