class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        def atMost(k):
            count = 0
            left = 0
            freq = defaultdict(int)
            distinct = 0
            for right in range(len(nums)):
                if freq[nums[right]] == 0:
                    distinct += 1
                freq[nums[right]] += 1

                while distinct > k:
                    freq[nums[left]] -= 1
                    if freq[nums[left]] == 0:
                        distinct -= 1
                    left += 1

                count += right - left + 1
            return count

        total_distinct = len(set(nums))
        return atMost(total_distinct) - atMost(total_distinct - 1)