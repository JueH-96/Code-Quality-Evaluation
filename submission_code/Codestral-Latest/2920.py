class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Dictionary to store the last occurrence of each number
        last_occurrence = defaultdict(int)
        # Dictionary to store the maximum gap between occurrences of each number
        max_gap = defaultdict(int)

        n = len(nums)

        # First pass: calculate the last occurrence of each number
        for i in range(n):
            if nums[i] in last_occurrence:
                max_gap[nums[i]] = max(max_gap[nums[i]], i - last_occurrence[nums[i]])
            last_occurrence[nums[i]] = i

        # Second pass: calculate the gap for numbers that wrap around the array
        for num in last_occurrence:
            max_gap[num] = max(max_gap[num], n - last_occurrence[num] + last_occurrence[num])

        # The minimum number of seconds needed is the minimum of the maximum gaps
        return min(max_gap.values()) // 2