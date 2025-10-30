from collections import deque

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        deq = deque()
        flip_effect = 0
        count = 0
        
        for i in range(n):
            if deq and deq[0] == i - 3:
                deq.popleft()
                flip_effect ^= 1
            
            state = nums[i] ^ flip_effect
            if state == 0:
                if i <= n - 3:
                    deq.append(i)
                    flip_effect ^= 1
                    count += 1
                else:
                    return -1
        
        return count