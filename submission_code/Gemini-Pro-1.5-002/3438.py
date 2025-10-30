class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        ans = []

        def count_peaks(arr, l, r):
            count = 0
            for i in range(l + 1, r):
                if arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
                    count += 1
            return count

        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                ans.append(count_peaks(nums, l, r))
            else:
                index, val = query[1], query[2]
                nums[index] = val

        return ans