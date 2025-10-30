class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_count = len(set(nums))
        freq = {}
        left = 0
        current_distinct = 0
        result = 0
        n = len(nums)

        for right in range(n):
            freq[nums[right]] = freq.get(nums[right], 0) + 1
            if freq[nums[right]] == 1:
                current_distinct += 1

            while current_distinct == distinct_count:
                # Add all subarrays that start at 'left' and extend to 'right' or beyond
                result += (n - right)
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    current_distinct -= 1
                left += 1

        return result