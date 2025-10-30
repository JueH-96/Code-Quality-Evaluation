class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        last_occurrence = [-1] * n
        
        for sec in range(1, m + 1):
            t = sec - 1
            idx = changeIndices[t] - 1
            last_occurrence[idx] = t
            
            if any(x == -1 for x in last_occurrence):
                continue
            
            mark_at = [-1] * sec
            for i in range(n):
                mark_at[last_occurrence[i]] = i
            
            credits = 0
            for t_inner in range(sec):
                if mark_at[t_inner] == -1:
                    credits += 1
                else:
                    need = nums[mark_at[t_inner]]
                    if need > credits:
                        break
                    credits -= need
            else:
                return sec
        
        return -1