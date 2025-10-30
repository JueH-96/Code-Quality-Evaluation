class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)

        def check(time):
            last_occurrence = {}
            for i in range(n):
                last_occurrence[i + 1] = -1

            for i in range(time):
                last_occurrence[changeIndices[i]] = i

            for i in range(1, n + 1):
                if last_occurrence[i] == -1:
                    return False

            available_seconds = 0
            marked = [False] * n
            seconds_used = 0

            for i in range(time):
                index = changeIndices[i]
                if i == last_occurrence[index]:
                    if available_seconds >= nums[index - 1]:
                        available_seconds -= nums[index - 1]
                        marked[index - 1] = True
                        seconds_used += nums[index - 1]
                    else:
                        return False
                else:
                    available_seconds += 1
            
            return all(marked)

        left = 1
        right = m
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1

        return ans