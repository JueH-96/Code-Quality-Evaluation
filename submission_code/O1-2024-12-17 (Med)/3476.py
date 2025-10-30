class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        # We want to make each element divisible by 3 using the fewest operations.
        # For any element x:
        #   - If x % 3 == 0, no operations are needed.
        #   - Otherwise, we need exactly 1 operation (either add/subtract 1).
        # Thus, the total number of operations is the count of elements that are not currently divisible by 3.
        
        return sum(1 for x in nums if x % 3 != 0)