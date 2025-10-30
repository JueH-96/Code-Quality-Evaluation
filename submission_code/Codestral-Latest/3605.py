class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        def find_min_value(target):
            for x in range(target):
                if (x | (x + 1)) == target:
                    return x
            return -1

        return [find_min_value(num) for num in nums]