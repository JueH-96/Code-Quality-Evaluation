from sortedcontainers import SortedList

class Solution:
    def maximumBeauty(self, nums: List[int], k: int) -> int:
        low, maxLen, window = [], [], 0
        
        for num in nums:
            low.append(num - k)
            window.append(num + k)
        
        sl, sr = SortedList(low), SortedList(window)
        
        for x, y in zip(sl, sr):
            if x <= y:
                maxLen.append(y - x + 1)
        
        for l in low:
            right = sr.bisect_right(l + 2 * k)
            left = sl.bisect_right(l)
            windowSize = right - left
            
            if (l + 2 * k) in sr and (l) in sl:
                windowSize -= 1 
            
            maxLen.append(windowSize)

        return max(maxLen)