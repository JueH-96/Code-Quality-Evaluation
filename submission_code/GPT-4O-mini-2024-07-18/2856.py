class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        total_distinct = len(set(nums))
        count = 0
        n = len(nums)

        for start in range(n):
            distinct_count = 0
            freq = defaultdict(int)

            for end in range(start, n):
                if freq[nums[end]] == 0:
                    distinct_count += 1
                freq[nums[end]] += 1

                if distinct_count == total_distinct:
                    count += 1

        return count