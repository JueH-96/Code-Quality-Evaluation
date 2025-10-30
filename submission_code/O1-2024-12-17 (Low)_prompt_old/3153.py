class Solution:
    def maxSum(self, nums: List[int], k: int) -> int:
        """
        We want to maximize the sum of squares of k elements after repeatedly applying the operation:
            nums[i], nums[j] = (nums[i] AND nums[j]), (nums[i] OR nums[j]).
        
        Key Insight:
        ----------------
        By performing these operations in the right sequence, we can effectively "merge" bits from different
        elements to form large values (via OR) while the counterpart elements become smaller (via AND). A crucial
        observation is that once a bit is used in forming a large OR value, it cannot be reused to form another
        equally large value unless that bit appears in multiple distinct elements from the start.

        In effect, each bit b that appears 'count[b]' times in the original array can appear at most 'count[b]'
        times among all final elements that contain bit b. Therefore, the problem reduces to:
        
        (1) Count how many elements have each bit set (0 <= bit < 31 for 1 <= nums[i] <= 10^9).
        (2) Greedily construct up to k final values in descending bit order:
            - At each step, form the largest possible value by including all bits b for which count[b] > 0.
            - Decrement count[b] by 1 for each such bit because we've "used" that bit once.
            - Repeat until we form k values or can no longer form a nonzero value.
        (3) The resulting set of up to k values is the maximum set (by individual value) we can form. 
            We then sum their squares.
        
        Example Walkthrough (nums = [2,6,5,8], k = 2):
        ------------------------------------------------
        nums bits:
          2 -> b1
          6 -> b1,b2
          5 -> b0,b2
          8 -> b3
        count of bits: b0=1, b1=2, b2=2, b3=1

        - First largest value:
            b3>0 => include bit3, b2>0 => include bit2, b1>0 => include bit1, b0>0 => include bit0
            => value = 15.  Decrement counts => b3=0, b2=1, b1=1, b0=0
        - Second largest value:
            b3=0 => skip, b2=1 => include, b1=1 => include, b0=0 => skip
            => value = 6.   Decrement => b2=0, b1=0
        Sum of squares = 15^2 + 6^2 = 225 + 36 = 261.

        This matches the example. We return the result modulo 10^9 + 7.
        """
        MOD = 10**9 + 7

        # Count how many elements have each bit set
        bit_count = [0]*31
        for x in nums:
            for b in range(31):
                if x & (1 << b):
                    bit_count[b] += 1

        ans = 0
        # Build up to k numbers, each time forming the largest possible number from available bits
        for _ in range(k):
            cur = 0
            for b in reversed(range(31)):
                if bit_count[b] > 0:
                    cur |= (1 << b)
                    bit_count[b] -= 1

            if cur == 0:
                # Can't form anything larger anymore
                break

            # Add cur^2 to answer
            ans = (ans + (cur * cur) % MOD) % MOD

        return ans