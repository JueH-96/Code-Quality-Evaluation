class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        m = n // 2
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1

        for count in counts.values():
            if count > 2:
                return False

        num_distinct = len(counts)
        if num_distinct < m:
            return False

        return True