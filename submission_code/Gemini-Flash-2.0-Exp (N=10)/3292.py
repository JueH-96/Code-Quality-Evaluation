class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        
        last_occurrence = {}
        for i in range(m):
            last_occurrence[changeIndices[i]] = i + 1
        
        if len(last_occurrence) < n:
            return -1
        
        def check(time):
            temp_nums = list(nums)
            marked = [False] * n
            
            for s in range(1, time + 1):
                idx = changeIndices[s-1]
                if temp_nums[idx-1] == 0 and not marked[idx-1]:
                    marked[idx-1] = True
                else:
                    temp_nums[idx-1] = max(0, temp_nums[idx-1] - 1)
            
            
            all_marked = True
            for i in range(n):
                if not marked[i]:
                    all_marked = False
                    break
            return all_marked
        
        left, right = 1, m
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if check(mid):
                ans = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return ans