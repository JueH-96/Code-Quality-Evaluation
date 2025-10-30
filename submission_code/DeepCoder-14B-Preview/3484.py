class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        delta = [t - n for t, n in zip(target, nums)]
        operations = 0
        for i in range(1, len(delta)):
            operations += abs(delta[i] - delta[i-1])
        operations += abs(delta[0]) if delta else 0
        return operations