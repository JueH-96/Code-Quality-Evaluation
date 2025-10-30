class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        freqs = list(counts.values())
        freqs.sort()
        
        distinct_count = len(counts)
        
        for freq in freqs:
            if freq > 1:
                cost = freq - 1
                if k >= cost:
                    k -= cost
                    distinct_count += 1
                else:
                    break
        
        
        return distinct_count