class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(index):
            if index == 0 or index == len(nums) - 1:
                return False
            return nums[index] > nums[index - 1] and nums[index] > nums[index + 1]

        def count_peaks_in_range(l, r):
            count = 0
            for i in range(l + 1, r):
                if is_peak(i):
                    count += 1
            return count

        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                result.append(count_peaks_in_range(l, r))
            elif query[0] == 2:
                index, val = query[1], query[2]
                nums[index] = val

        return result