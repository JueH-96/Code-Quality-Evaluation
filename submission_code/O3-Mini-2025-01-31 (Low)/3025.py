from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count frequency of powers of 2.
        cnt = [0] * 62  # to cover powers from 0 to 61 (target < 2^31, but splitting can require extra room)
        total = 0
        
        for num in nums:
            power = num.bit_length() - 1  # because num is power of 2
            cnt[power] += 1
            total += num
        
        # If even the total sum is less than target, it's impossible.
        if total < target:
            return -1
        
        ops = 0
        for power in range(61):
            # Check if the current bit in target is set.
            if (target >> power) & 1:
                # we need one unit of 2^power at least.
                if cnt[power] <= 0:
                    # find the next larger power that is available
                    j = power + 1
                    while j < 62 and cnt[j] <= 0:
                        j += 1
                    if j == 62:
                        return -1  # should not reach due to total sum check
                    # split the element from j down to power
                    while j > power:
                        cnt[j] -= 1
                        cnt[j - 1] += 2
                        ops += 1
                        j -= 1
                # Use one occurrence from this power (simulate using it to cover the bit)
                cnt[power] -= 1
            # After covering the requirement for the bit, we can combine any leftover lower bits
            # into a higher power: Every two occurrences of 2^power can form one 2^(power+1).
            if power < 61:
                cnt[power + 1] += cnt[power] // 2
        return ops