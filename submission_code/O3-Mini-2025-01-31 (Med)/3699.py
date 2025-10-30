from typing import List
import math
from collections import defaultdict

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        # Idea:
        # We want to count quadruples (p, q, r, s) (with p<q<r<s) that satisfy:
        #   nums[p] * nums[r] == nums[q] * nums[s]
        # and the index gaps: q-p > 1, r-q > 1, s-r > 1.
        #
        # An equivalent way to rewrite the equality is:
        #   nums[p] / nums[q] == nums[s] / nums[r]
        # (since nums[i]> 0 for all i).
        #
        # We can “split” the quadruple into a left pair (p,q) and a right pair (r,s):
        #   • left pair: indices p and q with p < q and q - p > 1.
        #   • right pair: indices r and s with r < s and s - r > 1.
        # Such that additionally the quadruple ordering
        #   p < q < r < s   and   r - q > 1  (i.e. r >= q+2)
        # holds.
        #
        # For a valid quadruple the ratio must match:
        #   (nums[p]/nums[q])  = (nums[s]/nums[r])
        #
        # So we “group” all valid left pairs by their reduced fraction: key = (A, B)
        # where A = nums[p]//gcd(nums[p], nums[q]) and B = nums[q]//gcd(nums[p], nums[q]).
        # For each left pair we only need to store the index q (the latter index)
        # since in the quadruple the left pair’s q must be less than the right pair’s r by at least 2.
        #
        # Similarly, for each valid right pair (r, s) (with s-r>1, i.e. s >= r+2)
        # we compute the reduced fraction for nums[s]/nums[r] (key = (nums[s]//g, nums[r]//g))
        # and store the index r (the first index in the right pair).
        #
        # Finally, for each common fraction key we combine left pairs and right pairs.
        # For a left pair with second index q and a right pair with first index r,
        # the overall quadruple is valid only if r >= q+2.
        #
        # Instead of doing a binary search for each left pair, we can sort both lists and
        # then count, with a two-pointer merge, for each left index how many right indices
        # are at least q+2.
        #
        # The overall complexity is O(n^2) (to build left/right groups) plus
        # the cost for merging counts (which is linear in the total number of pairs).
        #
        # Build groups of left pairs:
        left_pairs = defaultdict(list)  # key: (nums[p] reduced, nums[q] reduced); value: list of q indices
        n = len(nums)
        # For left pairs, choose indices p, q with q - p > 1 (i.e. p <= q-2).
        for q in range(2, n):
            for p in range(0, q - 1):  # p goes from 0 to q-2
                # (q-p>=2 guaranteed by p <= q-2)
                g = math.gcd(nums[p], nums[q])
                key = (nums[p] // g, nums[q] // g)
                left_pairs[key].append(q)
                
        # Build groups of right pairs:
        right_pairs = defaultdict(list)  # key: (nums[s] reduced, nums[r] reduced); value: list of r indices
        # For right pairs, choose indices r, s with s - r > 1 (i.e. s>= r+2).
        for r in range(0, n - 2):
            for s in range(r + 2, n):
                g = math.gcd(nums[s], nums[r])
                key = (nums[s] // g, nums[r] // g)
                right_pairs[key].append(r)
        
        total = 0
        # Now, for each fraction key that appears in both groups, combine the pairs.
        for key, leftList in left_pairs.items():
            if key in right_pairs:
                leftList.sort()  # leftList stores q indices (the second index from left pair)
                rightList = sorted(right_pairs[key])  # rightList stores r indices (the first index from right pair)
                
                # For a valid quadruple, we need r >= q + 2.
                # Use a two-pointer technique to count, for each q in leftList, how many r in rightList satisfy:
                #   r >= q + 2
                j = 0
                len_right = len(rightList)
                for q in leftList:
                    # Advance pointer until rightList[j] >= q+2.
                    while j < len_right and rightList[j] < q + 2:
                        j += 1
                    total += (len_right - j)
        return total

# ---------------------------
# The following main routine is provided for local testing.
# It is not needed when the solution is submitted in a contest platform.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.numberOfSubsequences([1, 2, 3, 4, 3, 6, 1]))   # Expected output: 1
    # Example 2:
    print(sol.numberOfSubsequences([3, 4, 3, 4, 3, 4, 3, 4])) # Expected output: 3