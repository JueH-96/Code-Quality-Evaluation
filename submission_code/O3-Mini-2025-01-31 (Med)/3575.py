from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # We'll use dynamic programming to compute the possible OR values for picking exactly t elements
        # from a prefix of nums, and similarly from a suffix (in reverse).
        #
        # left[i][t]: set of all possible OR values when picking exactly t elements from nums[0:i] (i is count)
        # right[i][t]: set of all possible OR values when picking exactly t elements from nums[i:n]
        #
        # Then, for every valid splitting index i (which means the first half comes from nums[0:i] and 
        # the second half comes from nums[i:n]), if left[i][k] and right[i][k] are both not empty, we try
        # all pairs of OR values and maximize (first_half OR) XOR (second_half OR).

        # Build left DP table
        # Create an array of length n+1, each element is a list of sets for counts 0..k.
        left = [ [set() for _ in range(k+1)] for _ in range(n+1) ]
        left[0][0].add(0)
        
        for i in range(1, n+1):
            num = nums[i-1]
            for t in range(k+1):
                # Option 1: do not pick current element
                left[i][t] = left[i][t].union(left[i-1][t])
                # Option 2: pick current element if possible
                if t > 0:
                    # For each OR value from previous state with one less element, update
                    for prev in left[i-1][t-1]:
                        left[i][t].add(prev | num)

        # Build right DP table
        right = [ [set() for _ in range(k+1)] for _ in range(n+1) ]
        right[n][0].add(0)
        for i in range(n-1, -1, -1):
            num = nums[i]
            for t in range(k+1):
                # Not picking nums[i]
                right[i][t] = right[i][t].union(right[i+1][t])
                if t > 0:
                    for prev in right[i+1][t-1]:
                        right[i][t].add(prev | num)

        # Now, consider every possible split index i (which implies
        # that the first k picks come from nums[0:i] and the second k picks come from nums[i:n])
        best = 0
        # valid splits: if left[i][k] and right[i][k] are nonempty.
        for i in range(n+1):
            if left[i][k] and right[i][k]:
                for a in left[i][k]:
                    for b in right[i][k]:
                        best = max(best, a ^ b)
        return best