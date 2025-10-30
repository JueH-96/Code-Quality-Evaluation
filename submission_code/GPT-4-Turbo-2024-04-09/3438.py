class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def count_peaks(l, r):
            count = 0
            for i in range(l + 1, r):
                if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                    count += 1
            return count
        
        results = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                results.append(count_peaks(l, r))
            elif query[0] == 2:
                index, value = query[1], query[2]
                nums[index] = value
        
        return results