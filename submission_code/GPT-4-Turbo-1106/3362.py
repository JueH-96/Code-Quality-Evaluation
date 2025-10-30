class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def count_distinct_subarrays(nums):
            n = len(nums)
            counts = []
            for i in range(n):
                seen = set()
                for j in range(i, n):
                    seen.add(nums[j])
                    counts.append(len(seen))
            counts.sort()
            return counts
        
        def find_median(counts):
            n = len(counts)
            if n % 2 == 1:
                return counts[n // 2]
            else:
                return min(counts[n // 2 - 1], counts[n // 2])
        
        counts = count_distinct_subarrays(nums)
        return find_median(counts)