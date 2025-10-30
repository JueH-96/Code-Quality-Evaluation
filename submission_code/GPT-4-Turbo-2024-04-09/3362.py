class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def median(arr):
            n = len(arr)
            arr.sort()
            if n % 2 == 1:
                return arr[n // 2]
            else:
                return arr[n // 2 - 1]
        
        n = len(nums)
        freq = defaultdict(int)
        results = []
        
        for start in range(n):
            freq.clear()
            unique_count = 0
            for end in range(start, n):
                if freq[nums[end]] == 0:
                    unique_count += 1
                freq[nums[end]] += 1
                results.append(unique_count)
        
        return median(results)