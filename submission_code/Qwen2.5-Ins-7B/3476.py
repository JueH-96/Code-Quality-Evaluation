from typing import List

class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        count = [0, 0, 0]
        for num in nums:
            count[num % 3] += 1
        
        if count[1] == 0 and count[2] == 0:
            return 0
        
        if count[1] == 0 or count[2] == 0:
            return min(count[1], count[2])
        
        if count[1] + count[2] <= count[0]:
            return count[1] + count[2]
        else:
            return count[0] + min((count[1] + count[2]) % 3, 3 - (count[1] + count[2]) % 3)