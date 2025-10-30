from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        freq = Counter(nums)
        max_len = 0
        
        # Special case: if the base x equals 1, note that 1^p = 1 for any power.
        # For a valid pattern with base 1 the elements will all be 1.
        # The pattern for r >= 0 has length 2·r+1, and it requires a total of:
        #   • if r == 0: 1 occurrence of 1,
        #   • if r >= 1: 2 occurrences of 1 for each level i = 0,..,r-1 plus 1 occurrence for the center (level r),
        #      i.e. a total of (2*r + 1) ones.
        if 1 in freq:
            cnt = freq[1]
            # We can only form a pattern whose length is odd.
            candidate = cnt if cnt % 2 == 1 else cnt - 1
            max_len = max(max_len, candidate)
        
        # Now, consider candidate bases x different from 1.
        # For any chosen base x, we try to build the symmetric pattern:
        # [x, x^2, x^4, ..., x^(2^(r-1)), x^(2^r), x^(2^(r-1)), ..., x^4, x^2, x]
        # Requirements:
        #   - For r = 0: simply need one occurrence of x.
        #   - For r >= 1: for each i in 0..r-1, we require that:
        #           freq[x^(2^i)] >= 2,
        #       and additionally, freq[x^(2^r)] >= 1 (the center element).
        for x in freq:
            if x == 1:
                continue  # already handled separately
            candidate = 1  # r=0 always gives a valid pattern [x]
            r = 1
            while True:
                valid = True
                # Check the "two copies" requirement for every level before the center.
                for i in range(r):
                    exp = 1 << i  # this computes 2^i
                    val = pow(x, exp)
                    if val > 10**9 or freq.get(val, 0) < 2:
                        valid = False
                        break
                if not valid:
                    break
                # Check center element: x^(2^r) needs at least one occurrence.
                center_exp = 1 << r
                center_val = pow(x, center_exp)
                if center_val > 10**9 or freq.get(center_val, 0) < 1:
                    break
                # If passes, we have a valid pattern of length (2*r + 1)
                candidate = 2 * r + 1
                r += 1
            max_len = max(max_len, candidate)
        
        return max_len

# Example test cases:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumLength([5, 4, 1, 2, 2]))  # Expected output: 3, since we can form [2, 4, 2]
    print(sol.maximumLength([1, 3, 2, 4]))     # Expected output: 1