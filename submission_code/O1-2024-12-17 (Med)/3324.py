class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        from collections import Counter
        
        n = len(nums)
        freq = Counter(nums)
        
        # d = number of distinct elements
        d = len(freq)
        
        # sum2 = sum of min(freq[x], 2) for each distinct element x
        sum2 = sum(min(count, 2) for count in freq.values())
        
        # We need each subset to have n/2 distinct elements (so d >= n/2)
        # and enough duplicates overall to fill 2*(n/2) = n distinct "slots" (so sum2 >= n).
        return d >= n // 2 and sum2 >= n