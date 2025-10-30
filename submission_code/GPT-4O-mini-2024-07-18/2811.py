class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        current = 1
        
        while len(result) < n:
            if (k - current) not in result:
                result.append(current)
            current += 1
        
        return sum(result)