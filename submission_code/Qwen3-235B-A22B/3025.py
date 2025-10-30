from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        count = [0] * 32  # To store counts of powers 2^0 to 2^31 (though target < 2^31)
        for num in nums:
            if num == 0:
                continue  # though problem states nums[i] > 0
            exp = num.bit_length() - 1
            if 0 <= exp < 32:
                count[exp] += 1
        
        total = 0
        for i in range(32):
            total += (1 << i) * count[i]
            if total >= target:
                break
        if total < target:
            return -1
        
        operations = 0
        for i in range(31):
            if (target >> i) & 1:
                if count[i] > 0:
                    count[i] -= 1
                else:
                    # Find the smallest j > i with count[j] > 0
                    j = i + 1
                    while j < 31 and count[j] == 0:
                        j += 1
                    if j >= 31:
                        return -1  # should not happen as total >= target
                    # Split from j down to i
                    for k in range(j, i, -1):
                        count[k] -= 1
                        count[k-1] += 2
                        operations += 1
                    # Now use one of i
                    count[i] -= 1  # after splitting, count[i] has at least 2
            # Merge pairs of i into i+1
            if i < 30:
                merged = count[i] // 2
                count[i+1] += merged
                count[i] %= 2
        
        return operations