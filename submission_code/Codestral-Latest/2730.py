class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Calculate the prefix OR and suffix OR arrays
        prefix_or = [0] * n
        suffix_or = [0] * n

        current_or = 0
        for i in range(n):
            current_or |= nums[i]
            prefix_or[i] = current_or

        current_or = 0
        for i in range(n-1, -1, -1):
            current_or |= nums[i]
            suffix_or[i] = current_or

        # Initialize the maximum OR value
        max_or = 0

        # Try all possible operations
        for i in range(n):
            # Calculate the OR value if we apply the operation on nums[i]
            current_or = (nums[i] << k) | (prefix_or[i-1] if i > 0 else 0) | (suffix_or[i+1] if i < n-1 else 0)
            max_or = max(max_or, current_or)

        return max_or