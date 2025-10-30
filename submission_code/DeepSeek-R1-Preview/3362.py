from collections import defaultdict
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        T = n * (n + 1) // 2
        k = (T - 1) // 2
        
        low = 1
        high = n
        answer = n
        
        while low <= high:
            mid = (low + high) // 2
            count = self.count_at_most(nums, mid)
            if count >= k + 1:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer
    
    def count_at_most(self, nums: List[int], K: int) -> int:
        count = 0
        freq = defaultdict(int)
        left = 0
        
        for right in range(len(nums)):
            freq[nums[right]] += 1
            
            while len(freq) > K:
                left_num = nums[left]
                freq[left_num] -= 1
                if freq[left_num] == 0:
                    del freq[left_num]
                left += 1
            
            count += right - left + 1
        
        return count