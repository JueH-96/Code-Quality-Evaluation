from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Initialize frequency of each power of 2
        powers = [0] * 31
        for num in nums:
            k = num.bit_length() - 1
            powers[k] += 1
        
        operations = 0
        # Iterate from highest power to lowest
        for k in range(30, -1, -1):
            # If current power is greater than target, break it down
            while k > 0 and (1 << k) > target:
                if powers[k] == 0:
                    break
                powers[k - 1] += 2 * powers[k]
                operations += powers[k]
                powers[k] = 0
            
            # If target needs this power
            if target & (1 << k):
                if powers[k] > 0:
                    powers[k] -= 1
                else:
                    # Find the nearest higher power to break down
                    m = k + 1
                    while m < 31 and powers[m] == 0:
                        m += 1
                    if m == 31:
                        return -1
                    # Break down from m to k
                    for level in range(m, k, -1):
                        powers[level - 1] += 2 * powers[level]
                        operations += powers[level]
                        powers[level] = 0
                    powers[k] -= 1
            # Carry over any excess to lower powers
            while k > 0 and powers[k] >= 2:
                powers[k - 1] += 2 * powers[k] // 2
                operations += powers[k] // 2
                powers[k] %= 2
        
        # Check if target can be formed
        current_sum = 0
        for k in range(31):
            current_sum += powers[k] * (1 << k)
            if current_sum > target:
                return -1
        if current_sum < target:
            return -1
        return operations