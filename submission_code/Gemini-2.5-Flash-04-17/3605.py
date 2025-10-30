from typing import List

class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        ans = []
        for target in nums:
            # Check if target can be represented as x OR (x+1).
            # Let's analyze the bitwise operation x OR (x+1).
            # If x is even, its binary representation ends in 0. x = ...b1b0, with b0=0. x+1 = ...b1(b0+1) = ...b11.
            # x OR (x+1) = (...b10) OR (...b11) = ...b11. The last bit is 1.
            # If x is odd, its binary representation ends in 1. x = ...b_k 0 1...1 (k ones), where b_k=0 is the LSB 0.
            # x+1 = ...b_k 1 0...0 (k zeros).
            # x OR (x+1) = (...b_k 0 1...1) OR (...b_k 1 0...0) = (...b_k 1 1...1) (k+1 ones). The last bit is 1 (if k >= 0).
            # So, for any non-negative integer x, x OR (x+1) is always odd.
            # Since nums[i] >= 2, x cannot be 0 (0 OR 1 = 1 != nums[i] >= 2).
            # If target is even, there is no non-negative integer x such that x OR (x+1) == target.
            if target % 2 == 0:
                ans.append(-1)
            else:
                # Target is odd. A solution x might exist.
                # Let target = x OR (x+1), and let k be the position of the Least Significant 0 bit of x (0-indexed from the right).
                # x has 0 at bit k, and 1s at bits 0 to k-1.
                # x+1 has 1 at bit k, and 0s at bits 0 to k-1.
                # x and x+1 have the same bits at positions greater than k.
                # x OR (x+1) will have 1s at bits 0 to k, and the same bits as x (and x+1) at positions greater than k.
                # This implies that the bits of target from position 0 up to k must all be 1s.
                # This means k must be less than the position of the first 0 bit from the right in target.
                
                # Let m be the number of trailing 1s in the binary representation of target.
                # If target = P 0 1...1 (m ones), then the first 0 bit is at position m.
                # If target is all 1s (e.g., 3, 7, 31), then m is the total number of bits, and there is no 0 bit.
                # In both cases, any k such that bits 0 through k of target are 1s must satisfy k < m.
                # The possible values for k (position of LSB 0 of x) are 0, 1, ..., m-1.
                
                # Find m, the length of the trailing sequence of 1s in target.
                m = 0
                temp = target
                while temp > 0 and temp & 1 == 1:
                    m += 1
                    temp >>= 1
                    
                # For each possible k in [0, m-1], we can construct a candidate value for x.
                # x_k has the same bits as target at positions > k, 0 at position k, and 1s at positions < k.
                # x_k = (bits of target above k) | (0 at bit k) | (1s at bits 0 to k-1)
                # bits of target above k: target & (~0 << (k+1)) which is (target >> (k+1)) << (k+1)
                # 1s at bits 0 to k-1: (1 << k) - 1
                # So, x_k = (target >> (k+1) << (k+1)) | ((1 << k) - 1)
                
                # We need the smallest x_k over possible k in [0, m-1].
                # The term (target >> (k+1) << (k+1)) is the higher part of x_k. As k increases, this part tends to decrease (or stay the same).
                # The term ((1 << k) - 1) is the lower part of x_k. As k increases, this part increases.
                # Since higher bits have more weight, the overall value x_k is minimized when the higher part is minimized.
                # This happens when k is maximized, i.e., k = m-1.
                
                k = m - 1
                
                # Calculate the minimum x, which corresponds to k = m-1.
                # Substitute k = m-1 into the formula:
                # x = (target >> ((m-1)+1) << ((m-1)+1)) | ((1 << (m-1)) - 1)
                # x = (target >> m << m) | ((1 << (m - 1)) - 1)
                
                # Note: The expression ((1 << (m - 1)) - 1) correctly calculates the mask of m-1 ones.
                # For m=1 (k=0), it is (1 << 0) - 1 = 1 - 1 = 0.
                # For m > 1 (k >= 1), it is the integer with m-1 bits set to 1.
                
                x = (target >> m << m) | ((1 << (m - 1)) - 1)
                    
                ans.append(x)

        return ans