from collections import Counter

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        counts = Counter(nums)
        operations = 0
        power_counts = Counter()
        for num in nums:
            power = 0
            val = num
            while val > 1:
                val >>= 1
                power += 1
            power_counts[power] += 1
            
        current_sum = 0
        for i in range(32):
            power_of_2 = 1 << i
            if (target >> i) & 1:
                needed_val = power_of_2
                if power_counts[i] > 0:
                    power_counts[i] -= 1
                    current_sum += needed_val
                else:
                    found_larger = False
                    for j in range(i + 1, 32):
                        if power_counts[j] > 0:
                            found_larger = True
                            for k in range(j, i, -1):
                                power_counts[k] -= 1
                                power_counts[k - 1] += 2
                                operations += 1
                            power_counts[i] -= 1
                            current_sum += needed_val
                            break
                    if not found_larger:
                        return -1
                        
        return operations