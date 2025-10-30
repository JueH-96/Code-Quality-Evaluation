class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        ans = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                count = 0
                for i in range(l + 1, r):
                    if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                        count += 1
                ans.append(count)
            else:
                index, val = query[1], query[2]
                nums[index] = val
        return ans