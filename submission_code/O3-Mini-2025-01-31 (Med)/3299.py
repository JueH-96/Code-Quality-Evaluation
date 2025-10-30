from collections import Counter
from typing import List

class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        # We first count the frequency of each number.
        freq = Counter(nums)
        ans = 0
        
        # Special case: when the base number is 1.
        # For base 1, note that all the elements in the pattern would be 1.
        # The valid pattern is [1] when m == 0, [1, 1, 1] when m == 1,
        # [1, 1, 1, 1, 1] when m == 2, etc.
        # In other words, the pattern length is 2*m + 1.
        # Given a frequency count freq[1] = cnt, the maximum odd number we can pick is:
        #    cnt if cnt is odd, else cnt - 1.
        if 1 in freq:
            cnt = freq[1]
            candidate = cnt if cnt % 2 == 1 else cnt - 1
            ans = max(ans, candidate)
        
        # For any other candidate base x (x != 1), the pattern is forced:
        # [x, x^2, x^4, ..., x^(2^m), ..., x^4, x^2, x] for some m >= 0.
        # For m == 0, the pattern is just [x] (which is valid if x exists).
        # For m >= 1, notice the pattern requires each term from level 0 up to m-1
        # to appear twice (once in the first half and once in the second half)
        # and the middle term x^(2^m) to appear at least once.
        #
        # Thus, for m >= 1, we require for each level i in [0, m-1]:
        #    freq.get(x^(2^i), 0) >= 2
        # and for level m:
        #    freq.get(x^(2^m), 0) >= 1.
        #
        # We try to “grow” the longest possible pattern for each candidate base.
        for x in freq:
            if x == 1:
                continue  # already handled
            # m = 0 always works because we only require one x.
            if freq[x] < 1:
                continue
            cur_max = 1  # pattern [x] of length 1 is always valid.
            m = 1  # start trying with m = 1, i.e. pattern: [x, x^2, x]
            # We will try increasing m until a requirement fails.
            while True:
                valid = True
                # Check each lower level from 0 to m-1 where two copies are required.
                for level in range(m):
                    exp = 1 << level  # this is 2**level
                    try:
                        v = pow(x, exp)
                    except OverflowError:
                        valid = False
                        break
                    # If the computed power exceeds the maximum allowed value, break.
                    if v > 10**9:
                        valid = False
                        break
                    if freq.get(v, 0) < 2:
                        valid = False
                        break
                if not valid:
                    break
                # Now check level m, the middle element, which only needs 1 copy.
                exp = 1 << m  # the exponent is 2**m
                try:
                    v = pow(x, exp)
                except OverflowError:
                    break
                if v > 10**9 or freq.get(v, 0) < 1:
                    break
                # If we reached here, a valid pattern of length 2*m+1 is possible.
                cur_max = 2*m + 1
                m += 1
            ans = max(ans, cur_max)
        
        return ans


# Sample test cases for local run.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # Input: [5,4,1,2,2]
    # Explanation: The subset {4,2,2} can be rearranged to [2,4,2] (with base 2) which satisfies:
    #   2, 2^2 (4), and 2 (mirror of the first element); pattern: [x, x^2, x] with m=1.
    # Expected output: 3
    print(sol.maximumLength([5,4,1,2,2]))  # Output: 3

    # Example 2:
    # Input: [1,3,2,4]
    # Only single-element patterns are possible.
    # Expected output: 1
    print(sol.maximumLength([1,3,2,4]))  # Output: 1