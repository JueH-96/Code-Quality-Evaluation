class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        result = []
        current = 1
        
        while len(result) < n:
            # Check if current number would create a forbidden pair
            complement = k - current
            
            # If complement is in our result array, we can't use current
            if complement in result:
                current += 1
                continue
            
            # Otherwise, we can safely add current to our result
            result.append(current)
            current += 1
        
        return sum(result)