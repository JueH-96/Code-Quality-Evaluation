class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # The required set is numbers 1 through k.
        required = set(range(1, k + 1))
        ops = 0
        
        # Traverse from the last element to the start
        for num in reversed(nums):
            ops += 1  # perform an operation (remove this element)
            # If this number is in the required set, remove it.
            if num in required:
                required.remove(num)
            # If we have collected all required numbers, return the operations count.
            if not required:
                return ops
        
        # Given the problem constraints, this line would never be hit
        return ops