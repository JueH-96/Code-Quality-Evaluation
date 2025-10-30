class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        current = 1
        
        while len(result) < n:
            # Check if current number can be added to the result
            # It should not form a pair with any existing number in result that sums to k
            can_add = True
            for num in result:
                if num + current == k:
                    can_add = False
                    break
            
            if can_add:
                result.append(current)
            
            current += 1
        
        return sum(result)