class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        if n == 1:
            return 1
        
        result = []
        current = 1
        
        while len(result) < n:
            valid = True
            for num in result:
                if num + current == k:
                    valid = False
                    break
            if valid:
                result.append(current)
            current += 1
        
        return sum(result)