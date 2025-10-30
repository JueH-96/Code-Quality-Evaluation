class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n, m = len(nums), len(changeIndices)
        
        def can_mark_all(seconds):
            last_occurrence = [0] * (n + 1)
            for i in range(seconds):
                last_occurrence[changeIndices[i]] = i + 1
            
            if 0 in last_occurrence[1:]:
                return False
            
            decrements = 0
            marked = [False] * (n + 1)
            
            for i in range(seconds - 1, -1, -1):
                idx = changeIndices[i]
                if i + 1 == last_occurrence[idx] and not marked[idx]:
                    if decrements < nums[idx - 1]:
                        return False
                    decrements -= nums[idx - 1]
                    marked[idx] = True
                else:
                    decrements += 1
            
            return all(marked[1:])
        
        left, right = 1, m
        while left < right:
            mid = (left + right) // 2
            if can_mark_all(mid):
                right = mid
            else:
                left = mid + 1
        
        return left if can_mark_all(left) else -1