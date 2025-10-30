from collections import Counter

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = Counter(nums)
        operations = 0
        current_sum = 0
        for i in range(32):
            if (target >> i) & 1:
                power_needed = 1 << i
                if counts[power_needed] > 0:
                    counts[power_needed] -= 1
                    current_sum += power_needed
                else:
                    j = i + 1
                    found_power = False
                    while j <= 31:
                        if counts[1 << j] > 0:
                            found_power = True
                            break
                        j += 1
                    if not found_power:
                        return -1
                    ops_needed = j - i
                    operations += ops_needed
                    counts[1 << j] -= 1
                    counts[power_needed] += 1
                    current_sum += power_needed
                    for k in range(j - 1, i - 1, -1):
                        counts[1 << k] += 1
        return operations