from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            return arr[i] > arr[i - 1] and arr[i] > arr[i + 1]

        def count_peaks_in_range(arr, l, r):
            count = 0
            for i in range(l + 1, r):
                if is_peak(arr, i):
                    count += 1
            return count

        answer = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                answer.append(count_peaks_in_range(nums, l, r))
            elif query[0] == 2:
                index, val = query[1], query[2]
                nums[index] = val

        return answer