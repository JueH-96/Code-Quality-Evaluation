from math import gcd
from functools import reduce
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Explanation:
        # In each allowed operation we choose two positive numbers a and b,
        # append a % b, and remove a and b. Notice that for any choices,
        #    gcd(a, b) == gcd(a % b, b)
        # so the greatest common divisor (call it g) of all numbers in the array
        # remains invariant.
        #
        # It turns out that an optimal strategy works in two phases.
        #
        # Phase 1. Convert all numbers except some “catalyst” copies of g into copies of g.
        #         More precisely, if at least one element is equal to g, by choosing
        #         the order carefully you can "fix" any other number a > g using g.
        #         (Because if you use g as the first operand and a (>g) as the second,
        #         then g % a = g, so you remove one copy of g and a, but then append g.)
        #         In this way every non‑g positive number can be removed (one by one)
        #         without “losing” the catalyst g.
        #
        # However, it might happen that initially no number equals g.
        # In that case one of the operations (on two non‑g numbers) can be arranged
        # so that the remainder equals g (this is always possible if g is the gcd
        # of the list). Once one g is present, the above idea works.
        #
        # Phase 2. Once only copies of g remain, you might still be able to “reduce” the
        #         array length further. But note: an operation on two copies of g gives
        #         g % g = 0. (Zero is positive? – It is not; only positive numbers may be used
        #         in an operation.)
        #
        # So if you have several copies of g, you must be careful.
        # Operations that “fix” a non‑g number (by pairing a g with a non‑g in the right order)
        # reduce the total array length by 1 but do not change the count of g.
        # But an operation that pairs two copies of g (in whichever order)
        # produces 0 (which is dead weight because zeros cannot be used for further operations)
        # and reduces the number of g’s by 2 (and increases the total length by 0 because
        # you remove 2 and add 1, a net reduction by 1).
        #
        # Thus after phase 1, if we let k be the number of copy of g in the array,
        # none of which we want to unnecessarily combine (because combining g with g, always
        # yields 0, and a zero cannot be used to “fix” any remaining number), the best we can do is:
        #
        #   • If initially no element equal to g was present (i.e. k==0), then one operation
        #     must be used to generate a copy of g; from then on you can convert everything.
        #     In that case the optimum is final length 1.
        #
        #   • Otherwise, after eliminating all non‑g’s you have k copies of g.
        #     If you then make operations on pairs of g’s (each such operation reducing length by 1)
        #     then:
        #         – If k is even you can pair them all, leaving k/2 zeros; final length is k/2.
        #         – If k is odd you can pair k-1 copies, leaving one g unpaired; final length is (k+1)//2.
        #
        # In summary, the answer is:
        #   • 1, if no number is initially equal to g,
        #   • or ceil(k/2) (i.e. (k+1)//2) if at least one number equals g.
        #
        # It can be shown (by carefully choosing the order of operations) that these final lengths
        # are achievable and in fact optimal.
        
        # Compute the gcd of all numbers.
        g = reduce(gcd, nums)
        
        # Count how many numbers equal g.
        count_g = 0
        for num in nums:
            if num == g:
                count_g += 1

        if count_g == 0:
            # If none equals g, we can create one g.
            return 1
        else:
            # When one or more g’s are present,
            # the optimal final length is ceil(count_g/2).
            return (count_g + 1) // 2

# Sample testing:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    print(sol.minimumArrayLength([1, 4, 3, 1]))   # Expected output: 1
    # Example 2
    print(sol.minimumArrayLength([5, 5, 5, 10, 5])) # Expected output: 2
    # Example 3
    print(sol.minimumArrayLength([2, 3, 4]))        # Expected output: 1

    # Additional tests:
    # All equal numbers:
    # For [5,5] → best to combine to zero giving final length 1.
    print(sol.minimumArrayLength([5, 5]))           # Expected output: 1
    # For [5,5,5] → optimal final length is ceil(3/2)=2.
    print(sol.minimumArrayLength([5, 5, 5]))        # Expected output: 2
    # A case with no initial g:
    # e.g. [4,6] with gcd = 2, but neither equals 2; optimal final length is 1.
    print(sol.minimumArrayLength([4, 6]))           # Expected output: 1