class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        nums_sorted = sorted(nums)
        for i, num in enumerate(nums_sorted):
            if num >= k:
                return i
        return len(nums)  # This line is a fallback as per problem constraints, but it's guaranteed not to be reached.