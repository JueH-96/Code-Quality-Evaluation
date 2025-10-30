class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        answer = []

        def is_peak(i):
            if i == 0 or i == n-1:
                return False
            return nums[i] > nums[i-1] and nums[i] > nums[i+1]

        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                count = 0
                for i in range(l, r+1):
                    if is_peak(i):
                        count += 1
                answer.append(count)
            else:
                index, val = query[1], query[2]
                nums[index] = val

        return answer