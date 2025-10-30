class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        from collections import defaultdict
        
        freq = defaultdict(int)
        for pos in nums:
            freq[pos] += 1
        
        for i in range(len(moveFrom)):
            from_pos = moveFrom[i]
            to_pos = moveTo[i]
            count = freq[from_pos]
            del freq[from_pos]
            freq[to_pos] += count
        
        return sorted(freq.keys())