from typing import List

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        low, high = 1, n
        res = 0
        
        while low <= high:
            mid = (low + high) // 2
            possible = False
            for left in range(n - mid + 1):
                right_window = left + mid - 1
                m_idx = left + (mid // 2)
                median = nums[m_idx]
                
                left_part_count = m_idx - left + 1
                sum_left = prefix[m_idx + 1] - prefix[left]
                part1 = median * left_part_count - sum_left
                
                sum_right = prefix[left + mid] - prefix[m_idx + 1]
                right_part_count = right_window - m_idx
                part2 = sum_right - median * right_part_count
                
                total = part1 + part2
                if total <= k:
                    possible = True
                    break
            
            if possible:
                res = mid
                low = mid + 1
            else:
                high = mid - 1
        
        return res