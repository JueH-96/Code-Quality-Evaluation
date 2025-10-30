class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        marked = [False] * n

        def can_mark(seconds):
            temp_nums = list(nums)
            operations = seconds
            marked_count = 0

            for s in range(1, min(seconds + 1, m + 1)):
                idx = changeIndices[s - 1] - 1
                if temp_nums[idx] > 0:
                    if operations > 0:
                        temp_nums[idx] -= 1
                        operations -= 1
                if temp_nums[idx] == 0:
                    if not marked[idx]:
                        marked[idx] = True
                        marked_count +=1
            
            for i in range(n):
                if not marked[i]:
                    ops_needed = temp_nums[i]
                    if ops_needed <= operations:
                        operations -= ops_needed
                        marked[i] = True
                        marked_count += 1

            return marked_count == n

        left = 1
        right = m
        ans = -1

        while left <= right:
            mid = (left + right) // 2
            marked = [False] * n
            if can_mark(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans