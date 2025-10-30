class Solution:
    def countInterestingSubarrays(self, nums: List[int], modulo: int, k: int) -> int:
        count = [0] * modulo
        count[0] = 1
        prefix = 0
        result = 0

        for num in nums:
            prefix = (prefix + num % modulo) % modulo
            result += count[prefix]
            count[prefix] += 1

        return result