from collections import defaultdict
from typing import List

class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        total_subarrays = n * (n + 1) // 2
        required = (total_subarrays - 1) // 2 + 1
        
        def count_at_most(k: int) -> int:
            if k == 0:
                return 0
            freq = defaultdict(int)
            left = 0
            distinct = 0
            result = 0
            for right in range(n):
                num = nums[right]
                if freq[num] == 0:
                    distinct += 1
                freq[num] += 1
                
                while distinct > k:
                    left_num = nums[left]
                    freq[left_num] -= 1
                    if freq[left_num] == 0:
                        distinct -= 1
                    left += 1
                
                result += (right - left + 1)
            return result
        
        low, high = 1, n
        answer = n
        while low <= high:
            mid = (low + high) // 2
            current_count = count_at_most(mid)
            if current_count >= required:
                answer = mid
                high = mid - 1
            else:
                low = mid + 1
        
        return answer