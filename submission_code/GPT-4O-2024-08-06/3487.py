class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        def is_subsequence(s, p):
            it = iter(s)
            return all(char in it for char in p)
        
        left, right = 0, len(targetIndices)
        max_operations = 0
        
        while left <= right:
            mid = (left + right) // 2
            temp_source = list(source)
            
            for i in range(mid):
                temp_source[targetIndices[i]] = ''
            
            if is_subsequence(''.join(temp_source), pattern):
                max_operations = mid
                left = mid + 1
            else:
                right = mid - 1
        
        return max_operations