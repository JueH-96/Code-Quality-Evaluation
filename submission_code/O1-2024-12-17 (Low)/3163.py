class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        total_sum = 0
        for i in range(n):
            freq = {}
            distinct_count = 0
            for j in range(i, n):
                freq[nums[j]] = freq.get(nums[j], 0) + 1
                if freq[nums[j]] == 1:
                    distinct_count += 1
                total_sum += distinct_count * distinct_count
        return total_sum