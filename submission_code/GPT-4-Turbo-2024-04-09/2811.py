class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        # Start with the smallest numbers and avoid pairs that sum to k
        result = []
        used = set()
        
        # Try to fill the array with the smallest possible numbers
        current = 1
        while len(result) < n:
            if (k - current) not in used and current not in used:
                result.append(current)
                used.add(current)
            current += 1
        
        # Calculate the sum of the array
        return sum(result)