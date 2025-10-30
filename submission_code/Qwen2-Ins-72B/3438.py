from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        peaks = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peaks[i] = 1

        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + peaks[i]

        results = []
        update = {}
        for q in queries:
            if q[0] == 1:
                l, r = q[1], q[2]
                if (l, r) in update:
                    results.append(update[(l, r)])
                else:
                    results.append(prefix_sum[r + 1] - prefix_sum[l])
            else:
                index, val = q[1], q[2]
                if index > 0 and index < n - 1:
                    if nums[index] > nums[index - 1] and nums[index] > nums[index + 1]:
                        peaks[index] -= 1
                        prefix_sum[index + 1] -= 1
                        prefix_sum[n] -= 1
                    if val > nums[index - 1] and val > nums[index + 1]:
                        peaks[index] += 1
                        prefix_sum[index + 1] += 1
                        prefix_sum[n] += 1
                nums[index] = val
                update[(q[1], q[2])] = prefix_sum[q[2] + 1] - prefix_sum[q[1]]
        return results