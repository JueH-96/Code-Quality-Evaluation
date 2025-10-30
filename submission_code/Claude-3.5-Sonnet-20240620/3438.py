class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(i):
            return nums[i-1] < nums[i] > nums[i+1]

        def update_peaks(index):
            for i in range(max(1, index-1), min(len(nums)-1, index+2)):
                peaks[i] = 1 if is_peak(i) else 0

        peaks = [0] * len(nums)
        for i in range(1, len(nums) - 1):
            peaks[i] = 1 if is_peak(i) else 0

        answer = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                answer.append(sum(peaks[l+1:r]))
            else:
                index, val = query[1], query[2]
                nums[index] = val
                update_peaks(index)

        return answer