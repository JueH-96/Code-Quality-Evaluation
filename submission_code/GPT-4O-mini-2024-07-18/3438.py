class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(i):
            return 1 <= i < len(nums) - 1 and nums[i] > nums[i - 1] and nums[i] > nums[i + 1]

        def count_peaks(l, r):
            return sum(1 for i in range(l + 1, r) if is_peak(i))

        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                result.append(count_peaks(l, r))
            elif query[0] == 2:
                index, val = query[1], query[2]
                nums[index] = val
        
        return result