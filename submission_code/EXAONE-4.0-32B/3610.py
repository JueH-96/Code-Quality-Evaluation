from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        result = []
        for i in range(n - k + 1):
            window = nums[i:i+k]
            freq_map = {}
            for num in window:
                freq_map[num] = freq_map.get(num, 0) + 1
            
            items = []
            for num, count in freq_map.items():
                items.append((count, num))
            
            items.sort(key=lambda item: (-item[0], -item[1]))
            
            total = 0
            top_x = items[:x]
            for count, num in top_x:
                total += count * num
                
            result.append(total)
            
        return result