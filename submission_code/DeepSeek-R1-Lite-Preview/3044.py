class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        target = set(range(1, k+1))
        collected = set()
        operations = 0
        for num in reversed(nums):
            collected.add(num)
            operations += 1
            if collected.issuperset(target):
                return operations
        return len(nums)  # As per problem constraints, this line is redundant but ensures completeness