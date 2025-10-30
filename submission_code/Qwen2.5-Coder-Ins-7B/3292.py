class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        marked = [False] * n
        last_occurrence = [-1] * (n + 1)
        
        for s in range(1, m + 1):
            last_occurrence[changeIndices[s]] = s
        
        def can_mark_all_by_second(k):
            remaining = nums[:]
            marked = [False] * n
            for s in range(1, k + 1):
                if last_occurrence[changeIndices[s]] == s:
                    if remaining[changeIndices[s] - 1] > 0:
                        remaining[changeIndices[s] - 1] -= 1
                    else:
                        marked[changeIndices[s] - 1] = True
                else:
                    remaining[changeIndices[s] - 1] -= 1
            
            return all(marked)
        
        left, right = 1, m
        while left <= right:
            mid = (left + right) // 2
            if can_mark_all_by_second(mid):
                right = mid - 1
            else:
                left = mid + 1
        
        return left if left <= m else -1