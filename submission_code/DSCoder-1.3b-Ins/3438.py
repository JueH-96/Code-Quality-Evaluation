class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        result = []
        for query in queries:
            l, r = query[1], query[2]
            if l == 0 or r == len(nums) - 1:
                result.append(0)
            elif nums[l-1] > nums[l] and nums[l] > nums[r]:
                result.append(1)
            else:
                result.append(0)
        return result