class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            if i > 0 and i < len(arr) - 1:
                return arr[i] > arr[i - 1] and arr[i] > arr[i + 1]
            return False

        def count_peaks_in_range(arr, l, r):
            count = 0
            for i in range(l, r + 1):
                if is_peak(arr, i):
                    count += 1
            return count

        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                result.append(count_peaks_in_range(nums, l, r))
            else:
                index, val = query[1], query[2]
                nums[index] = val
        return result