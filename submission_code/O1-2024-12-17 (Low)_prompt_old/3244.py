class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # If there is only one or two elements, we can never end up with
        # more than one element after performing operations:
        #
        #  - n = 1: No operation can be performed, so the answer is 1.
        #  - n = 2: A single operation takes the two positives, inserts
        #    their remainder, and removes them.  That leaves exactly 1 element.
        #
        # For n >= 3, the key observation (as shown by the examples and
        # by trying out various cases) is that the final length is always
        # either 1 or 2.  The deciding factor is whether we can ever produce
        # a "nontrivial" remainder from some pair.  Concretely:
        #
        #   - If for EVERY number x in nums, x is a multiple of the minimum
        #     element m, then whenever we pick two positive elements a,b with
        #     (say) a >= b, we get a % b = 0 if b divides a exactly.  Because
        #     they are all multiples of a common m, we keep getting 0 as the
        #     remainder (or the smaller number again if we do b % a in the
        #     wrong order, but that does not help create new, smaller, nonzero
        #     values).  In effect, we cannot "break" the array down to a single
        #     survivor, and we get stuck with 2 leftover elements in the end.
        #
        #   - Otherwise (there is at least one pair a,b where the smaller does
        #     NOT divide the larger), we can eventually produce a smaller
        #     positive remainder.  Repeating such remainders in a chain of
        #     operations can merge all positives down to a single leftover
        #     element (the final array length is 1).
        #
        # Thus the rule for n >= 3 becomes:
        #   - Let m = min(nums).
        #   - If every number in nums is a multiple of m, return 2.
        #   - Else return 1.
        #
        # Combine all cases:
        #
        #   1) If len(nums) <= 2, answer is 1.
        #   2) Otherwise, check if every x % m == 0. If so, answer = 2, else 1.

        n = len(nums)
        if n <= 2:
            return 1

        m = min(nums)
        for x in nums:
            if x % m != 0:
                return 1
        return 2