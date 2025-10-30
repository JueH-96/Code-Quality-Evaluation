class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        for f, t in zip(moveFrom, moveTo):
            if f == t:
                continue
            if f in counts:
                cnt = counts.pop(f)
                counts[t] = counts.get(t, 0) + cnt
        
        return sorted(counts.keys())