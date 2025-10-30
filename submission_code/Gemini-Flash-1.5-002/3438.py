class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        ans = []
        for q in queries:
            if q[0] == 2:
                nums[q[1]] = q[2]
            else:
                l, r = q[1], q[2]
                count = 0
                for i in range(l + 1, r):
                    if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                        count += 1
                ans.append(count)
        return ans