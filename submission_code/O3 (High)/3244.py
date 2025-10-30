from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        """
        Let m be the minimum element in the array and c its frequency.

        • Any non-zero remainder that ever gets produced is necessarily m.
          (If we divide by something ≥ m the remainder is < divisor, but m is the
          smallest positive number, so the only positive remainder possible is m.)

        • Therefore an operation that keeps the remainder non-zero never
          decreases the amount of m’s – it removes one m but inserts another.

        • To actually get rid of m’s we have to pair two of them together; that
          remainder is 0.  
          Such an operation cuts the amount of m’s by 2 and puts one 0 into the
          array.

        • First we can pair every m with a *different* non-minimum element,
          consuming all non-minimum numbers without reducing the number of m’s.
          After that only the c copies of m are left.  From then on we must use
          ⌊c/2⌋ operations of the form (m,m)→0, each shrinking the count of m’s
          by 2 and leaving a 0 behind.  If c is odd one last m stays untouched.

        Hence the final array will consist of  
            – ⌈c/2⌉ elements if c copies of m were present:
                •   ⌊c/2⌋ zeros
                •   plus one extra m when c is odd

        Consequently the minimum possible length is ⌈c/2⌉, i.e.
        (c + 1) // 2 using integer arithmetic.
        """
        # count how many times the minimum value occurs
        mn = min(nums)
        cnt_min = nums.count(mn)
        # ⌈cnt_min / 2⌉
        return (cnt_min + 1) // 2