class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if min(nums) < k:
            return -1
        distinct_above_k = set()
        for num in nums:
            if num > k:
                distinct_above_k.add(num)
        return len(distinct_above_k)