from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [None] * n  # Store the peaks with None
        count = 0

        for i in range(1, n - 1):
            if nums[i - 1] < nums[i] > nums[i + 1]:
                peaks[i] = True
                count += 1
            else:
                peaks[i] = False

        ans = []
        modification = {}  # Store the modifications

        for t, a, b in queries:
            if t == 2:
                modification[a] = b
            else:
                prev_left, prev_right = a - 1, b + 1
                count -= peaks[a] + (peaks[a + 1] if a + 1 < n else False)
                while modification.get(prev_left, None) is not None and prev_left >= 0:
                    prev_left -= 1
                while modification.get(prev_right, None) is not None and prev_right < n:
                    prev_right += 1

                to_check = [a, a + 1 if a + 1 < n else a]
                count += sum(nums[curr - 1] < modification[curr] > nums[curr + 1] for curr in to_check if (curr - 1) >= 0 and (curr + 1) < n)

                for i in range(a, b + 1):
                    if i in modification:
                        nums[i] = modification[i]

                ans.append(count)
        return ans