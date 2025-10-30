from typing import List
import math
from functools import reduce

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # In this problem an operation chooses two positive numbers a and b,
        # appends a % b to the end, and removes a and b.
        # Thus every operation reduces the total length by one.
        # However, once a zero is created it never can participate in later operations (since we require > 0).
        #
        # A key invariant is that if you look at all positive values in the array,
        # their greatest common divisor (gcd) stays invariant – in fact, it becomes exactly the gcd
        # of the initial array.
        #
        # One can show that if the overall gcd is 1, then by applying the Euclidean algorithm
        # (which is what a % b essentially does) cleverly one may eventually “combine”
        # the positives to get a 1. And once you have a 1 (and another positive) you can eventually reduce
        # everything down to a single element (by repeated operations using 1 to “cancel out” other numbers).
        #
        # On the other hand, if the overall gcd g > 1 then every positive number remains a multiple of g.
        # In that case an operation on two identical positive numbers yields 0:
        #    Example: [5, 5] -> operation (5,5) produces 5 % 5 = 0; array becomes [0].
        # And 0’s cannot be used in further operations.
        #
        # For g > 1, the optimal strategy is as follows:
        #   • If the array length is 1, nothing can be done; answer is 1.
        #   • If the array length is 2, one operation on the two numbers yields a single positive result;
        #     answer is 1.
        #   • For arrays of length >= 3, one may always reduce the array to exactly 2 elements.
        #     (This can be argued by pairing off most of the positive numbers into operations
        #      that yield zero, while one “survives”.)
        #
        # Thus the answer is:
        #   - 1 if the overall gcd is 1.
        #   - Otherwise, if there is only one number or two numbers then answer is 1.
        #     For arrays with >= 3 numbers, answer is 2.
        #
        # Compute the overall gcd of the array:
        g = reduce(math.gcd, nums)
        if g == 1:
            return 1
        
        n = len(nums)
        # For arrays of length 1 or 2, one operation can bring it to one positive.
        if n <= 2:
            return 1
        # Otherwise, the best we can do is to reduce the array to 2 numbers.
        return 2

# Example usage and simple tests:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumArrayLength([1, 4, 3, 1]))  # Expected output: 1
    print(sol.minimumArrayLength([5, 5, 5, 10, 5]))  # Expected output: 2
    print(sol.minimumArrayLength([2, 3, 4]))  # Expected output: 1
    # Additional tests:
    print(sol.minimumArrayLength([5, 5]))     # Expected output: 1 (pairing yields one positive)
    print(sol.minimumArrayLength([5, 5, 5]))  # Expected output: 2