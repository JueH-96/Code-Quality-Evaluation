class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:
        def max_steps_to_zero(x):
            steps = 0
            while x > 0:
                x = x // 4
                steps += 1
            return steps
        
        total_operations = 0
        for l, r in queries:
            max_steps = 0
            # We only need to consider the maximum number of steps needed for any number in the range [l, r]
            # because we can always pair the largest number with any other number to reduce the total steps.
            for num in range(l, r + 1):
                max_steps = max(max_steps, max_steps_to_zero(num))
            total_operations += max_steps
        
        return total_operations

# Example usage:
sol = Solution()
print(sol.minOperations([[1,2],[2,4]]))  # Output: 3
print(sol.minOperations([[2,6]]))        # Output: 4