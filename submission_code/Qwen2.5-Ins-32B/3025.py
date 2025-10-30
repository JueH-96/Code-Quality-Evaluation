from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        if sum(nums) < target:
            return -1
        
        count = [0] * 31
        for num in nums:
            count[int(math.log2(num))] += 1
        
        operations = 0
        for i in range(30):
            if target & (1 << i):
                if count[i] > 0:
                    count[i] -= 1
                else:
                    j = i
                    while j < 30 and count[j] == 0:
                        j += 1
                    if j == 30:
                        return -1
                    count[j] -= 1
                    operations += j - i
                    while j > i:
                        count[j - 1] += 2
                        j -= 1
            count[i + 1] += count[i] // 2
        
        return operations