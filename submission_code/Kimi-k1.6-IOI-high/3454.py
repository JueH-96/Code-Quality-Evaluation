class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        d = [t - n for t, n in zip(target, nums)]
        pos = [max(0, num) for num in d]
        neg = [max(0, -num) for num in d]
        
        def brush_strokes(arr):
            res = 0
            prev = 0
            for num in arr:
                if num > prev:
                    res += num - prev
                prev = num
            return res
        
        return brush_strokes(pos) + brush_strokes(neg)