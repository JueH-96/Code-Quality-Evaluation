from typing import List

class Solution:
    def countArrays(self, original: List[int], bounds: List[List[int]]) -> int:
        n = len(original)
        # Compute prefix sums of differences (offsets)
        offsets = [0] * n
        for i in range(1, n):
            offsets[i] = offsets[i-1] + (original[i] - original[i-1])
        
        # Find the feasible range for copy[0]
        low = -10**18  # start with a very low number
        high = 10**18  # start with a very high number
        for i in range(n):
            # For copy[i] = copy[0] + offsets[i], we have:
            # bounds[i][0] <= copy[0] + offsets[i] <= bounds[i][1]
            # => copy[0] in [bounds[i][0] - offsets[i], bounds[i][1] - offsets[i]]
            low = max(low, bounds[i][0] - offsets[i])
            high = min(high, bounds[i][1] - offsets[i])
        
        # Number of valid arrays is the count of valid copy[0] values
        if low > high:
            return 0
        return high - low + 1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    original = [1, 2, 3, 4]
    bounds = [[1, 2], [2, 3], [3, 4], [4, 5]]
    print(sol.countArrays(original, bounds))  # Output: 2
    
    # Example 2:
    original = [1,2,3,4]
    bounds = [[1,10], [2,9], [3,8], [4,7]]
    print(sol.countArrays(original, bounds))  # Output: 4
    
    # Example 3:
    original = [1,2,1,2]
    bounds = [[1,1], [2,3], [3,3], [2,3]]
    print(sol.countArrays(original, bounds))  # Output: 0