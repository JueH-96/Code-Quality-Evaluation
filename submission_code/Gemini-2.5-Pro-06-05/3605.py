from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        """
        Constructs an array ans such that for each index i, ans[i] | (ans[i] + 1) == nums[i],
        and each ans[i] is minimized.

        Args:
            nums: A list of prime integers.

        Returns:
            A list ans satisfying the conditions.
        """
        ans = []
        for num in nums:
            # The bitwise OR of any integer x and x+1 is always odd.
            # The only even prime is 2. Thus, for num=2, no solution exists.
            if num == 2:
                ans.append(-1)
                continue

            # For any odd prime num, a solution for x always exists.
            # We need to find the minimal non-negative x s.t. x | (x+1) == num.
            
            # An even solution, x = num - 1, always exists because (num-1) | num = num for any odd num.
            # An odd solution for x exists only if num % 4 == 3.
            # If an odd solution exists, it is smaller than the even one.
            
            if num % 4 == 1:
                # No odd solution exists. The minimal x is the even one.
                ans.append(num - 1)
            else:  # num % 4 == 3
                # An odd solution exists and is smaller.
                # The minimal x is num - 2^(m-1), where m is the number of trailing 1s in num's binary form.
                # m equals the number of trailing 0s in (num+1)'s binary representation (v_2(num+1)).
                y_plus_1 = num + 1
                
                # m can be found by finding the index of the least significant bit of num+1.
                # The bitwise operation (n & -n) isolates the LSB.
                # .bit_length() - 1 gives its zero-based index.
                m = (y_plus_1 & -y_plus_1).bit_length() - 1
                
                # The minimal answer is num - 2^(m-1).
                # 1 << (m - 1) is an efficient way to calculate 2^(m-1).
                ans.append(num - (1 << (m - 1)))
                
        return ans