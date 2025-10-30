class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        left = 0
        right = len(targetIndices)
        ans = 0
        while left <= right:
            mid = (left + right) // 2
            temp_source = list(source)
            removed = set(targetIndices[:mid])
            k=0
            for i in range(n):
                if i in removed:
                    temp_source[i] = ''
            
            new_source = "".join(temp_source)
            
            i = 0
            j = 0
            while i < len(new_source) and j < m:
                if new_source[i] == pattern[j]:
                    j += 1
                i += 1
            
            if j == m:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans