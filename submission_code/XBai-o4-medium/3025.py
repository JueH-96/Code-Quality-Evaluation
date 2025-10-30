from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = [0] * 31
        for num in nums:
            ex = num.bit_length() - 1
            counts[ex] += 1
        
        carry = 0
        operations = 0
        
        for i in range(31):
            available = counts[i] + carry
            need = (target >> i) & 1
            
            if need:
                if available >= 1:
                    available -= 1
                else:
                    # Find the first j > i with counts[j] > 0
                    j = i + 1
                    found = -1
                    while j < 31:
                        if counts[j] > 0:
                            found = j
                            break
                        j += 1
                    if found == -1:
                        return -1
                    # Split found down to i
                    operations += (found - i)
                    # Add 2^(found - i) to available
                    available += 1 << (found - i)
                    available -= 1  # use one
                    counts[found] -= 1
            # Update carry for next iteration
            carry = available // 2
        
        return operations